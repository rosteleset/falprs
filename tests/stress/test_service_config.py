"""Tests of the service config rendering and the implementation settings."""

import tomllib
from pathlib import Path

import pytest
import yaml

import service
from common import EnvironmentFailure, parse_duration
from service import load_template, render_config, retention_of, workflow_config


@pytest.fixture(scope="module", params=["rust", "cpp"])
def impl(request):
    return request.param


@pytest.fixture(scope="module")
def template(impl):
    return load_template(impl)


def test_template_differs_from_functional_config(template):
    assert {logger["level"] for logger, _ in service.loggers(template).values()} == {"info"}
    assert retention_of(template) == {
        "log_faces_ttl": "10m", "log_faces_cleanup_interval": "1m",
        "events_ttl": "10m", "events_cleanup_interval": "1m",
        "log_barcodes_ttl": "10m", "log_barcodes_cleanup_interval": "1m",
        "events_log_ttl": "10m", "events_log_cleanup_interval": "1m",
        "failed_ttl": "10m",
    }
    frs, lprs = workflow_config(template, "frs"), workflow_config(template, "lprs")
    assert "pipelines-per-cpu" not in frs and "pipelines-per-cpu" not in lprs


def test_templates_hold_the_same_settings():
    rust, cpp = load_template("rust"), load_template("cpp")
    assert retention_of(rust) == retention_of(cpp)
    assert service.api_port(rust) == service.api_port(cpp)
    assert service.log_files(rust) == service.log_files(cpp)
    assert {s: d[k] for s, (d, k) in service.databases(rust).items()} == \
        {s: d[k] for s, (d, k) in service.databases(cpp).items()}
    for subsystem, keys in (("frs", ("screenshots-path", "events-path", "barcodes-path")),
                            ("lprs", ("screenshots-path", "failed-path"))):
        for key in keys:
            assert workflow_config(rust, subsystem)[key] == workflow_config(cpp, subsystem)[key]


def test_screenshot_dirs_match_template(template):
    frs, lprs = workflow_config(template, "frs"), workflow_config(template, "lprs")
    for key in ("screenshots-path", "events-path", "barcodes-path"):
        assert Path(frs[key]).is_relative_to(service.SCREENSHOT_DIRS["frs"])
    for key in ("screenshots-path", "failed-path"):
        assert Path(lprs[key]).is_relative_to(service.SCREENSHOT_DIRS["lprs"])


def test_render_keeps_template_by_default(template, impl):
    config = render_config(template, impl=impl)
    assert retention_of(config) == retention_of(template)
    assert "pipelines-per-cpu" not in workflow_config(config, "frs")
    # The template itself is not changed
    assert config is not template


def test_retention_overrides(template, impl):
    config = render_config(template, impl=impl,
                           retention={"log_faces_ttl": "5m", "events_log_cleanup_interval": "30s",
                                      "failed_ttl": None})
    effective = retention_of(config)
    assert effective["log_faces_ttl"] == "5m"
    assert effective["events_log_cleanup_interval"] == "30s"
    assert effective["failed_ttl"] == "10m"
    assert retention_of(template)["log_faces_ttl"] == "10m"
    key = "events-log-maintenance-interval" if impl == "cpp" else "events-log-cleanup-interval"
    assert workflow_config(config, "lprs")[key] == "30s"


def test_invalid_retention(template, impl):
    with pytest.raises(EnvironmentFailure):
        render_config(template, impl=impl, retention={"events_ttl": "1.5m"})


def test_pipelines_per_cpu_in_both_subsystems():
    config = render_config(load_template("rust"), impl="rust", pipelines_per_cpu=16)
    assert workflow_config(config, "frs")["pipelines-per-cpu"] == 16
    assert workflow_config(config, "lprs")["pipelines-per-cpu"] == 16


def test_pipelines_per_cpu_rejected_under_cpp():
    template = load_template("cpp")
    with pytest.raises(EnvironmentFailure):
        render_config(template, impl="cpp", pipelines_per_cpu=16)
    # Without the flag the C++ config renders
    assert "pipelines-per-cpu" not in workflow_config(render_config(template, impl="cpp"), "frs")


def test_pipelines_per_cpu_positive():
    with pytest.raises(EnvironmentFailure):
        render_config(load_template("rust"), impl="rust", pipelines_per_cpu=0)


def test_pg_port_and_log_dir(template, impl):
    config = render_config(template, impl=impl, port=5433, log_dir="/tmp/test_falprs/logs/frs-ppc8")
    dsns = {subsystem: database[key] for subsystem, (database, key) in service.databases(config).items()}
    assert dsns["frs"].endswith("@localhost:5433/test_frs")
    assert dsns["lprs"].endswith("@localhost:5433/test_lprs")
    files = service.log_files(config)
    assert files["frs"] == Path("/tmp/test_falprs/logs/frs-ppc8/frs-log.txt")
    assert files["default"] == Path("/tmp/test_falprs/logs/frs-ppc8/log.txt")


def test_written_config_reloads(tmp_path):
    config = render_config(load_template("rust"), impl="rust", pipelines_per_cpu=8)
    path = service.write_config(config, tmp_path / "c.toml")
    assert tomllib.loads(path.read_text()) == config
    config = render_config(load_template("cpp"), impl="cpp")
    path = service.write_config(config, tmp_path / "c.yaml")
    assert yaml.safe_load(path.read_text()) == config


def test_default_config_path_follows_the_format():
    assert service.config_suffix(load_template("rust")) == ".toml"
    assert service.config_suffix(load_template("cpp")) == ".yaml"


def test_implementation_defaults():
    impl, binary = service.implementation({})
    assert impl == "rust" and binary == (service.REPO_DIR / "rust/target/release/falprs").resolve()
    with pytest.raises(EnvironmentFailure):
        service.implementation({"FALPRS_IMPL": "cpp"})
    impl, binary = service.implementation({"FALPRS_IMPL": "cpp", "FALPRS_BIN": "/opt/cpp/falprs"})
    assert impl == "cpp" and binary == Path("/opt/cpp/falprs")
    _, binary = service.implementation({"FALPRS_BIN": "/opt/x/falprs"})
    assert binary == Path("/opt/x/falprs")
    with pytest.raises(EnvironmentFailure):
        service.implementation({"FALPRS_IMPL": "go"})
    with pytest.raises(EnvironmentFailure):
        service.check_binary(Path("/nonexistent/falprs"))
    assert service.pg_port({}) == 5432 and service.pg_port({"pg_port": "6432"}) == 6432


def test_service_cpus():
    assert 1 <= service.service_cpus() <= len(__import__("os").sched_getaffinity(0))


def test_parse_duration():
    assert parse_duration("250ms") == 0.25
    assert parse_duration("0") == 0
    assert parse_duration("2m") == 120
    assert parse_duration("4h") == 4 * 3600
    assert parse_duration("1.5s") == 1.5
    assert parse_duration("5w") is None
