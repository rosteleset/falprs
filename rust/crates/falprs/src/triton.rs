//! Triton Inference Server gRPC client.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tonic::transport::{Channel, Endpoint};

mod proto {
    tonic::include_proto!("inference");
}

use proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use proto::ModelInferRequest;

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
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    pub fn as_f32(&self) -> Vec<f32> {
        self.data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect()
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

#[derive(Default)]
pub struct Triton {
    channels: Mutex<HashMap<String, Channel>>,
}

impl Triton {
    fn channel(&self, endpoint: &str) -> Result<Channel> {
        let mut channels = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(channel) = channels.get(endpoint) {
            return Ok(channel.clone());
        }
        let channel = Endpoint::from_shared(format!("http://{endpoint}"))
            .with_context(|| format!("invalid inference server address {endpoint}"))?
            .connect_lazy();
        channels.insert(endpoint.to_string(), channel.clone());
        Ok(channel)
    }

    pub async fn infer(&self, request: InferRequest<'_>) -> Result<Vec<Tensor>> {
        let mut client = GrpcInferenceServiceClient::new(self.channel(request.endpoint)?)
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX);
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
        let mut grpc_request = tonic::Request::new(ModelInferRequest {
            model_name: request.model_name.to_string(),
            inputs,
            outputs,
            raw_input_contents,
            ..Default::default()
        });
        grpc_request.set_timeout(request.timeout);

        let response = tokio::time::timeout(request.timeout, client.model_infer(grpc_request))
            .await
            .context("inference timed out")??
            .into_inner();
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
