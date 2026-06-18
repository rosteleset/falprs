#undef linux

#include <userver/clients/dns/component.hpp>
#include <userver/clients/http/component_list.hpp>
#include <userver/components/fs_cache.hpp>
#include <userver/components/minimal_server_component_list.hpp>
#include <userver/server/handlers/http_handler_static.hpp>
#include <userver/server/handlers/log_level.hpp>
#include <userver/server/handlers/on_log_rotate.hpp>
#include <userver/server/handlers/ping.hpp>
#include <userver/testsuite/testsuite_support.hpp>
#include <userver/utils/daemon_run.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <string_view>

// clang-format off
#ifdef BUILD_LPRS
  #include "lprs_api.hpp"
#endif

#ifdef BUILD_FRS
  #include "frs_api.hpp"
#endif
// clang-format on

int main(const int argc, char* argv[])
{
  namespace po = boost::program_options;

  auto desc = userver::utils::BaseRunOptions();
  desc.add_options()("version,v", "print version")(
      "config,c", po::value<std::string>()->required(), "path to server config");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.contains("help")) {
      std::cerr << desc << std::endl;
      return 0;
    }
    if (vm.contains("version")) {
      std::cout << FALPRS_VERSION << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    std::cerr << "Try --help for usage" << std::endl;
    return 1;
  }

  // clang-format off
  const auto component_list = userver::components::MinimalServerComponentList()
    .Append<userver::server::handlers::Ping>()

#ifdef BUILD_LPRS
    .Append<Lprs::Api>()
    .Append<Lprs::GroupsCache>()
    .Append<Lprs::VStreamsConfigCache>()
    .Append<Lprs::VStreamGroupCache>()
    .Append<Lprs::Workflow>()
    .Append<userver::components::Postgres>(std::string(Lprs::Workflow::kDatabase).data())
#endif

#ifdef BUILD_FRS
    .Append<Frs::Api>()
    .Append<Frs::GroupsCache>()
    .Append<Frs::ConfigCache>()
    .Append<Frs::VStreamsConfigCache>()
    .Append<Frs::FaceDescriptorCache>()
    .Append<Frs::VStreamDescriptorsCache>()
    .Append<Frs::SGConfigCache>()
    .Append<Frs::SGDescriptorsCache>()
    .Append<Frs::Workflow>()
    .Append<userver::components::Postgres>(std::string(Frs::Workflow::kDatabase).data())
#endif

    .Append<userver::components::FsCache>("fs-cache-main")
    .Append<userver::server::handlers::HttpHandlerStatic>()
    .Append<userver::server::handlers::LogLevel>()
    .Append<userver::server::handlers::OnLogRotate>()
    .AppendComponentList(userver::clients::http::ComponentList())
    .Append<userver::components::TestsuiteSupport>()
    .Append<userver::clients::dns::Component>();
  // clang-format on

  return userver::utils::DaemonMain(vm, component_list);
}
