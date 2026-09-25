#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <userver/components/component_base.hpp>
#include <userver/ugrpc/client/client_factory_component.hpp>
#include <userver/yaml_config/merge_schemas.hpp>

struct TritonTensor
{
  std::string name;
  std::string datatype;
  std::vector<int64_t> shape;
  std::string data;  // raw bytes
};

struct TritonInferenceRequest
{
  std::string model_name;
  std::string model_version;
  std::vector<TritonTensor> inputs;
  std::vector<std::string> requested_outputs;
  std::chrono::milliseconds timeout;
  std::string endpoint;
};

struct TritonInferenceResponse
{
  std::vector<TritonTensor> outputs;
};

template <typename T>
TritonTensor MakeTensor(std::string name,
  std::string datatype,
  std::vector<int64_t> shape,
  const T* data,
  size_t byte_count)
{
  return TritonTensor{std::move(name), std::move(datatype), std::move(shape),
    std::string(reinterpret_cast<const char*>(data), byte_count)};
}

class TritonClientService final : public userver::components::ComponentBase
{
public:
  static constexpr std::string_view kName = "triton-client";

  TritonClientService(const userver::components::ComponentConfig& config,
    const userver::components::ComponentContext& context);

  ~TritonClientService() override = default;

  // Coroutine-safe; blocks the calling coroutine until gRPC completes.
  // Throws std::runtime_error on gRPC error.
  TritonInferenceResponse Infer(const TritonInferenceRequest& request);

  static userver::yaml_config::Schema GetStaticConfigSchema();

private:
  userver::ugrpc::client::ClientFactory& client_factory_;
};
