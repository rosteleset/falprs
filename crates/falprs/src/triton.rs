//! Triton Inference Server gRPC client: tonic's channels, with the inference requests framed by hand.

use std::collections::HashMap;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use http_body::{Frame, SizeHint};
use prost::Message;
use prost::encoding::{WireType, encode_key, encode_varint, encoded_len_varint, key_len};
use tonic::codec::{Codec, Streaming};
use tonic::codegen::Service;
use tonic::codegen::http::{self, HeaderValue, header};
use tonic::transport::{Channel, Endpoint};

// Generated from the Triton protos, whose names the lints have no say over; only the inference messages are used
#[allow(clippy::all, dead_code)]
mod proto {
    tonic::include_proto!("inference");
}

use proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use proto::{ModelInferRequest, ModelInferResponse};

/// A tensor with its contents in Triton's raw little-endian layout.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: Bytes,
}

/// Floats in little-endian byte order, viewed as their bytes.
struct LittleEndianFloats(Vec<f32>);

impl AsRef<[u8]> for LittleEndianFloats {
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Tensor {
    /// Takes the floats over as the tensor's bytes, without copying them on little-endian targets: a
    /// copy of a detector input (about 5 MB) into a new zero-filled buffer cost a memset, a copy and,
    /// for a buffer the allocator took fresh from the kernel, a page fault per page, on every frame.
    pub fn fp32(name: &str, shape: Vec<i64>, mut values: Vec<f32>) -> Self {
        if cfg!(target_endian = "big") {
            for v in &mut values {
                *v = f32::from_bits(v.to_bits().to_le());
            }
        }
        Self {
            name: name.to_string(),
            datatype: "FP32".to_string(),
            shape,
            data: Bytes::from_owner(LittleEndianFloats(values)),
        }
    }

    pub fn as_f32(&self) -> Vec<f32> {
        self.data.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect()
    }
}

pub struct InferRequest<'a> {
    /// `host:port` of the server's gRPC endpoint.
    pub endpoint: &'a str,
    pub model_name: &'a str,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<&'a str>,
    pub timeout: Duration,
}

/// HTTP/2 connections per inference server. A connection error fails every request in flight on that
/// connection, so with several only their share fails. Each connection is also driven by one task, which
/// with the CPU saturated waits for its turn among the pipeline tasks, so more connections send the large
/// input tensors more often. Under an overload of 400 FRS + 400 LPRS streams the error rate was 35.9 %
/// with 1 connection, 23.9 % with 4, 17.6 % with 8 and 17.3 % with 16; without overload the throughput
/// was the same for all of them. One connection task used half a core at most, so it is not core bound.
const CONNECTIONS_PER_ENDPOINT: usize = 8;

/// The connections to one inference server, used in turn.
struct Connections {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl Connections {
    fn new(endpoint: &str) -> Result<Self> {
        let endpoint = Endpoint::from_shared(format!("http://{endpoint}")).with_context(|| format!("invalid inference server address {endpoint}"))?;
        // Every lazily connected channel opens its own connection
        let channels = (0..CONNECTIONS_PER_ENDPOINT).map(|_| endpoint.connect_lazy()).collect();
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    fn next(&self) -> Channel {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        self.channels[i].clone()
    }
}

#[derive(Default)]
pub struct Triton {
    connections: Mutex<HashMap<String, Arc<Connections>>>,
}

impl Triton {
    fn connections(&self, endpoint: &str) -> Result<Arc<Connections>> {
        let mut connections = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = connections.get(endpoint) {
            return Ok(c.clone());
        }
        let c = Arc::new(Connections::new(endpoint)?);
        connections.insert(endpoint.to_string(), c.clone());
        Ok(c)
    }

    /// Runs an inference within the request's timeout. A request that fails because its connection failed,
    /// e.g. closed by h2 with a PROTOCOL_ERROR on frames of another stream or for too many reset streams, is
    /// sent once more on the next connection, within the same timeout: inference has no side effects.
    pub async fn infer(&self, request: InferRequest<'_>) -> Result<Vec<Tensor>> {
        let connections = self.connections(request.endpoint)?;
        let mut raw_input_contents = Vec::with_capacity(request.inputs.len());
        let inputs = request
            .inputs
            .into_iter()
            .map(|tensor| {
                raw_input_contents.push(tensor.data);
                InferInputTensor {
                    name: tensor.name,
                    datatype: tensor.datatype,
                    shape: tensor.shape,
                    ..Default::default()
                }
            })
            .collect();
        let outputs = request
            .outputs
            .iter()
            .map(|name| InferRequestedOutputTensor {
                name: name.to_string(),
                ..Default::default()
            })
            .collect();
        let infer_request = ModelInferRequest {
            model_name: request.model_name.to_string(),
            inputs,
            outputs,
            ..Default::default()
        };
        // The chunks are `Bytes`, so the body of the second attempt shares them
        let chunks = grpc_frame(&infer_request, &raw_input_contents)?;

        let response = tokio::time::timeout(request.timeout, async {
            match model_infer(connections.next(), chunks.clone()).await {
                Err(status) if is_connection_error(&status) => model_infer(connections.next(), chunks).await,
                result => result,
            }
        })
        .await
        .context("inference timed out")??;
        if response.outputs.len() > response.raw_output_contents.len() {
            bail!("inference response has fewer raw contents than outputs");
        }
        Ok(response
            .outputs
            .into_iter()
            .zip(response.raw_output_contents)
            .map(|(output, data)| Tensor {
                name: output.name,
                datatype: output.datatype,
                shape: output.shape,
                data,
            })
            .collect())
    }
}

/// Field number of `ModelInferRequest.raw_input_contents`, the last field of the message.
const RAW_INPUT_CONTENTS_FIELD: u32 = 7;

/// The gRPC message of `request` with `raw_input_contents` appended, as chunks of the request body: the
/// length-prefixed message up to the first content, then each content and the field key and length of the
/// next. The contents are not copied: prost, through tonic's encoder, copied them into one buffer of the
/// whole message, a few megabytes per frame, fresh from the allocator each time. Prost writes the fields in
/// field-number order and the contents are the last field, so the bytes are the same as prost's.
fn grpc_frame(request: &ModelInferRequest, contents: &[Bytes]) -> Result<Vec<Bytes>> {
    debug_assert!(request.raw_input_contents.is_empty());
    let content_prefix_len = |c: &Bytes| key_len(RAW_INPUT_CONTENTS_FIELD) + encoded_len_varint(c.len() as u64);
    let message_len = request.encoded_len() + contents.iter().map(|c| content_prefix_len(c) + c.len()).sum::<usize>();
    let Ok(message_len) = u32::try_from(message_len) else {
        bail!("inference request of {message_len} bytes is too large for gRPC");
    };
    let mut chunks = Vec::with_capacity(2 * contents.len() + 1);
    let mut head = BytesMut::with_capacity(5 + request.encoded_len() + contents.first().map_or(0, content_prefix_len));
    // Not compressed, then the message length in big-endian
    head.put_u8(0);
    head.put_u32(message_len);
    request.encode(&mut head).expect("the buffer has the encoded length");
    for content in contents {
        encode_key(RAW_INPUT_CONTENTS_FIELD, WireType::LengthDelimited, &mut head);
        encode_varint(content.len() as u64, &mut head);
        chunks.push(head.split().freeze());
        chunks.push(content.clone());
    }
    if !head.is_empty() {
        chunks.push(head.freeze());
    }
    Ok(chunks)
}

/// A request body of ready chunks.
struct Chunks {
    chunks: std::vec::IntoIter<Bytes>,
    remaining: u64,
}

impl http_body::Body for Chunks {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(mut self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        let chunk = self.chunks.next();
        if let Some(chunk) = &chunk {
            self.remaining -= chunk.len() as u64;
        }
        Poll::Ready(chunk.map(|chunk| Ok(Frame::data(chunk))))
    }

    fn is_end_stream(&self) -> bool {
        self.remaining == 0
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::with_exact(self.remaining)
    }
}

/// `ModelInfer` with a body made of `chunks`, handled as tonic's generated client handles a unary call.
///
/// The timeout stays on the client: a request that runs out of time is cancelled with RST_STREAM. It is not
/// sent as `grpc-timeout`, because when that deadline expires on Triton while the input is still arriving,
/// Triton ends the stream twice, with DEADLINE_EXCEEDED and then with INTERNAL "Unable to parse request",
/// and h2 closes the whole connection on the second trailers.
async fn model_infer(mut channel: Channel, chunks: Vec<Bytes>) -> Result<ModelInferResponse, tonic::Status> {
    std::future::poll_fn(|cx| channel.poll_ready(cx))
        .await
        .map_err(|e| tonic::Status::unknown(format!("Service was not ready: {e}")))?;
    let remaining = chunks.iter().map(|c| c.len() as u64).sum();
    let body = Chunks {
        chunks: chunks.into_iter(),
        remaining,
    };
    let mut request = http::Request::new(tonic::body::Body::new(body));
    *request.method_mut() = http::Method::POST;
    *request.uri_mut() = http::Uri::from_static("/inference.GRPCInferenceService/ModelInfer");
    *request.version_mut() = http::Version::HTTP_2;
    request.headers_mut().insert(header::TE, HeaderValue::from_static("trailers"));
    request.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/grpc"));

    let response = channel.call(request).await.map_err(|e| tonic::Status::from_error(e.into()))?;
    // A trailers-only response carries its status in the headers
    if let Some(status) = tonic::Status::from_header_map(response.headers()) {
        if status.code() != tonic::Code::Ok {
            return Err(status);
        }
        return Err(tonic::Status::internal("Missing response message."));
    }
    let status_code = response.status();
    let decoder = tonic_prost::ProstCodec::<ModelInferRequest, ModelInferResponse>::default().decoder();
    // No size limit, as before with the generated client: tonic's default of 4 MB fails the outputs of the
    // vehicle footprint detector on large frames
    let mut body = Streaming::new_response(decoder, response.into_body(), status_code, None, Some(usize::MAX));
    let message = body.message().await?.ok_or_else(|| tonic::Status::internal("Missing response message."))?;
    // An error status in the trailers fails the call, as in tonic
    body.trailers().await?;
    Ok(message)
}

/// Whether a request failed because its HTTP/2 connection failed or was closed, not because of the request:
/// a GOAWAY (sent or received), an I/O error, a stream the server refused before processing it, or the
/// connection closed while the request waited for its response.
fn is_connection_error(status: &tonic::Status) -> bool {
    let mut source: Option<&(dyn StdError + 'static)> = status.source();
    while let Some(error) = source {
        if let Some(error) = error.downcast_ref::<h2::Error>() {
            return error.is_go_away() || error.is_io() || error.reason() == Some(h2::Reason::REFUSED_STREAM);
        }
        if let Some(error) = error.downcast_ref::<hyper::Error>()
            && error.is_canceled()
        {
            return true;
        }
        source = error.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_tensors_hold_little_endian_floats() {
        let values = [1.0f32, -2.5, f32::MIN_POSITIVE, 0.1];
        let tensor = Tensor::fp32("input", vec![1, 4], values.to_vec());
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(tensor.data, expected);
        assert_eq!(tensor.as_f32(), values);
    }

    #[test]
    fn grpc_frames_are_prosts_encoding() {
        let request = ModelInferRequest {
            model_name: "scrfd".to_string(),
            inputs: vec![InferInputTensor {
                name: "input.1".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 3, 2, 2],
                ..Default::default()
            }],
            outputs: vec![InferRequestedOutputTensor {
                name: "448".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        // Lengths with one- and two-byte varints, and an empty content
        let contents = [Bytes::from(vec![7u8; 48]), Bytes::from(vec![9u8; 300]), Bytes::new()];
        let framed: Vec<u8> = grpc_frame(&request, &contents).unwrap().concat();
        let mut whole = request.clone();
        whole.raw_input_contents = contents.iter().map(|c| c.to_vec()).collect();
        let message = whole.encode_to_vec();
        assert_eq!(framed[0], 0);
        assert_eq!(u32::from_be_bytes(framed[1..5].try_into().unwrap()) as usize, message.len());
        assert_eq!(framed[5..], message[..]);
    }

    #[test]
    fn connection_errors_are_told_from_request_errors() {
        let refused = h2::Error::from(h2::Reason::REFUSED_STREAM);
        assert!(is_connection_error(&tonic::Status::from_error(Box::new(refused))));
        let reset = h2::Error::from(h2::Reason::INTERNAL_ERROR);
        assert!(!is_connection_error(&tonic::Status::from_error(Box::new(reset))));
        assert!(!is_connection_error(&tonic::Status::unavailable("model is not ready")));
    }
}
