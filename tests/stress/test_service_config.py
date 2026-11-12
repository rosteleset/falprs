"""Tests of the service config rendering and the implementation settings."""

from pathlib import Path

import pytest
import yaml

import service
from common import EnvironmentFailure, parse_duration
from service import load_template, render_config, retention_of, workflow_config


@pytest.fixture(scope="module")
def template():
    return load_template()


def loggers(config):
    return service.components(config)["logging"]["loggers"]


def test_template_differs_from_functional_config(template):
    assert {logger["level"] for logger in loggers(template).values()} == {"info"}
    frs, lprs = workflow_config(template, "frs"), workflow_config(template, "lprs")
    assert frs["log-faces-ttl"] == "10m" and frs["clear-old-log-faces"] == "1m"
    assert frs["events-ttl"] == "10m" and frs["clear-old-events"] == "1m"
    assert frs["log-barcodes-ttl"] == "10m" and frs["clear-old-log-barcodes"] == "1m"
    assert lprs["events-log-ttl"] == "10m" and lprs["events-log-maintenance-interval"] == "1m"
    assert lprs["failed-ttl"] == "10m"
    assert "pipelines-per-cpu" not in frs and "pipelines-per-cpu" not in lprs


def test_screenshot_dirs_match_template(template):
    frs, lprs = workflow_config(template, "frs"), workflow_config(template, "lprs")
    for key in ("screenshots-path", "events-path", "barcodes-path"):
        assert Path(frs[key]).is_relative_to(service.SCREENSHOT_DIRS["frs"])
    for key in ("screenshots-path", "failed-path"):
        assert Path(lprs[key]).is_relative_to(service.SCREENSHOT_DIRS["lprs"])


def test_render_keeps_template_by_default(template):
    config = render_config(template, impl="rust")
    assert retention_of(config) == retention_of(template)
    assert "pipelines-per-cpu" not in workflow_config(config, "frs")
    # The template itself is not changed
    assert config is not template


def test_retention_overrides(template):
    config = render_config(template, impl="rust",
                           retention={"log_faces_ttl": "5m", "events_log_maintenance_interval": "30s",
                                      "failed_ttl": None})
    assert workflow_config(config, "frs")["log-faces-ttl"] == "5m"
    assert workflow_config(config, "lprs")["events-log-maintenance-interval"] == "30s"
    assert workflow_config(config, "lprs")["failed-ttl"] == "10m"
    assert workflow_config(template, "frs")["log-faces-ttl"] == "10m"


def test_invalid_retention(template):
    with pytest.raises(EnvironmentFailure):
        render_config(template, impl="rust", retention={"events_ttl": "1.5m"})


def test_pipelines_per_cpu_in_both_workflows(template):
    config = render_config(template, impl="rust", pipelines_per_cpu=16)
    assert workflow_config(config, "frs")["pipelines-per-cpu"] == 16
    assert workflow_config(config, "lprs")["pipelines-per-cpu"] == 16


def test_pipelines_per_cpu_rejected_under_cpp(template):
    with pytest.raises(EnvironmentFailure):
        render_config(template, impl="cpp", pipelines_per_cpu=16)
    # Without the flag the C++ config renders
    assert "pipelines-per-cpu" not in workflow_config(render_config(template, impl="cpp"), "frs")


def test_pipelines_per_cpu_positive(template):
    with pytest.raises(EnvironmentFailure):
        render_config(template, impl="rust", pipelines_per_cpu=0)


def test_pg_port_and_log_dir(template):
    config = render_config(template, impl="rust", port=5433, log_dir="/tmp/test_falprs/logs/frs-ppc8")
    parts = service.components(config)
    assert parts["frs-postgresql-database"]["dbconnection"].endswith("@localhost:5433/test_frs")
    assert parts["lprs-postgresql-database"]["dbconnection"].endswith("@localhost:5433/test_lprs")
    assert loggers(config)["frs"]["file_path"] == "/tmp/test_falprs/logs/frs-ppc8/frs-log.txt"
    assert loggers(config)["default"]["file_path"] == "/tmp/test_falprs/logs/frs-ppc8/log.txt"


def test_written_config_reloads(template, tmp_path):
    config = render_config(template, impl="rust", pipelines_per_cpu=8)
    path = service.write_config(config, tmp_path / "c.yaml")
    assert yaml.safe_load(path.read_text()) == config


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
