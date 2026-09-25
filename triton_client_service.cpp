#include "triton_client_service.hpp"

#include "grpc_service.pb.h"
#include "grpc_service_client.usrv.pb.hpp"

#include <userver/ugrpc/client/call_options.hpp>
#include <userver/ugrpc/client/client_factory_component.hpp>
#include <userver/yaml_config/merge_schemas.hpp>

namespace inference
{
  class GRPCInferenceServiceClient;
}
TritonClientService::TritonClientService(const userver::components::ComponentConfig& config,
  const userver::components::ComponentContext& context)
  : ComponentBase(config, context),
    client_factory_(context.FindComponent<userver::ugrpc::client::ClientFactoryComponent>().GetFactory())
{
}

TritonInferenceResponse TritonClientService::Infer(const TritonInferenceRequest& request)
{
  auto client = client_factory_.MakeClient<inference::GRPCInferenceServiceClient>(
    "triton-" + request.model_name, request.endpoint);

  inference::ModelInferRequest proto_request;
  proto_request.set_model_name(request.model_name);
  proto_request.set_model_version(request.model_version);

  for (const auto& [name, datatype, shape, data] : request.inputs)
  {
    auto* input = proto_request.add_inputs();
    input->set_name(name);
    input->set_datatype(datatype);
    for (const auto dim : shape)
    {
      input->add_shape(dim);
    }
    proto_request.add_raw_input_contents(data);
  }

  for (const auto& output_name : request.requested_outputs)
  {
    auto* output = proto_request.add_outputs();
    output->set_name(output_name);
  }

  userver::ugrpc::client::CallOptions call_options;
  call_options.SetTimeout(request.timeout);

  auto response = client.ModelInfer(proto_request, std::move(call_options));

  TritonInferenceResponse result;
  const auto& raw_contents = response.raw_output_contents();
  const auto& outputs = response.outputs();

  result.outputs.reserve(static_cast<size_t>(outputs.size()));
  for (int i = 0; i < outputs.size(); ++i)
  {
    TritonTensor tensor;
    tensor.name = outputs[i].name();
    tensor.datatype = outputs[i].datatype();
    for (const auto dim : outputs[i].shape())
    {
      tensor.shape.push_back(dim);
    }
    if (i < raw_contents.size())
    {
      tensor.data = raw_contents[i];
    }
    result.outputs.push_back(std::move(tensor));
  }

  return result;
}

userver::yaml_config::Schema TritonClientService::GetStaticConfigSchema()
{
  return userver::yaml_config::MergeSchemas<ComponentBase>(R"~(
# yaml
type: object
description: Triton gRPC inference client service
additionalProperties: false
properties: {}
)~");
}
