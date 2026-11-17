//! Triton Inference Server gRPC client.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use tonic::transport::{Channel, Endpoint};

// Generated from the Triton protos, whose names the lints have no say over
#[allow(clippy::all)]
mod proto {
    tonic::include_proto!("inference");
}

use proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use proto::{ModelInferRequest, ModelInferResponse};

/// A tensor with its contents in Triton's raw little-endian layout.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn fp32(name: &str, shape: Vec<i64>, values: &[f32]) -> Self {
        Self {
            name: name.to_string(),
            datatype: "FP32".to_string(),
            shape,
            data: {
                // Whole 4-byte chunks, which the compiler copies in bulk on little-endian targets
                let mut data = vec![0; values.len() * 4];
                for (bytes, v) in data.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                    *bytes = v.to_le_bytes();
                }
                data
            },
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
                raw_input_contents.push(Bytes::from(tensor.data));
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
        // The raw contents are `Bytes`, so a copy of the request for the second attempt shares them
        let infer_request = ModelInferRequest {
            model_name: request.model_name.to_string(),
            inputs,
            outputs,
            raw_input_contents,
            ..Default::default()
        };

        let response = tokio::time::timeout(request.timeout, async {
            match model_infer(connections.next(), infer_request.clone()).await {
                Err(status) if is_connection_error(&status) => model_infer(connections.next(), infer_request).await,
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

/// The timeout stays on the client: a request that runs out of time is cancelled with RST_STREAM. It is not
/// sent as `grpc-timeout`, because when that deadline expires on Triton while the input is still arriving,
/// Triton ends the stream twice, with DEADLINE_EXCEEDED and then with INTERNAL "Unable to parse request",
/// and h2 closes the whole connection on the second trailers.
async fn model_infer(channel: Channel, request: ModelInferRequest) -> Result<ModelInferResponse, tonic::Status> {
    let mut client = GrpcInferenceServiceClient::new(channel)
        .max_decoding_message_size(usize::MAX)
        .max_encoding_message_size(usize::MAX);
    Ok(client.model_infer(request).await?.into_inner())
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
        let tensor = Tensor::fp32("input", vec![1, 4], &values);
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(tensor.data, expected);
        assert_eq!(tensor.as_f32(), values);
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
