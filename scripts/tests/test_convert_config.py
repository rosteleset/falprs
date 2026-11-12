"""Tests of scripts/convert_config.py. Run with: python3 -m pytest scripts/tests

The golden files in fixtures/ are the converter's output of the old example and test configs; the
Rust unit tests check that they mean the same as config.toml.example and tests/test_config.toml.
Regenerate them with UPDATE_GOLDEN=1 after a deliberate change of the output.
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = SCRIPTS_DIR.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONVERTER = SCRIPTS_DIR / "convert_config.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import convert_config  # noqa: E402

MINIMAL = """
components_manager:
    components:
        server:
            listener:
                port: 9051
            listener-monitor:
                port: 9052
        logging:
            loggers:
                default:
                    file_path: '/var/log/falprs/log.txt'
        lprs-postgresql-database:
            dbconnection: 'postgresql://u:p@h/lprs'
        frs-postgresql-database:
            dbconnection: 'postgresql://u:p@h/frs'
"""


def convert(text):
    toml, warnings = convert_config.convert_text(text)
    return tomllib.loads(toml), warnings


@pytest.mark.parametrize(
    "source, golden, name",
    [
        (FIXTURES / "config.yaml.example", FIXTURES / "config.converted.toml", "config.yaml"),
        (REPO_DIR / "tests/test_config.yaml", FIXTURES / "test_config.converted.toml", "test_config.yaml"),
    ],
)
def test_old_configs_convert_to_the_golden_files(source, golden, name):
    toml, warnings = convert_config.convert_text(source.read_text(), name)
    assert warnings == []
    if os.environ.get("UPDATE_GOLDEN"):
        golden.write_text(toml)
    assert toml == golden.read_text()


def test_settings_left_out_keep_the_values_used_before():
    config, warnings = convert(MINIMAL)
    assert warnings == []
    assert config["server"]["max-request-size"] == 1024 * 1024
    # The LPRS default of allow-group-id-without-auth was 0, the new one is 1
    assert config["lprs"]["no-auth-group-id"] == 0
    assert config["frs"]["no-auth-group-id"] == 1
    for subsystem in ("lprs", "frs"):
        database = config[subsystem]["database"]
        assert database["max-pool-size"] == 16
        assert database["network-timeout"] == "500ms"
        assert database["statement-timeout"] == "250ms"
        assert "discovery-timeout" not in database
    assert config["logging"]["default"] == {"file": "/var/log/falprs/log.txt", "level": "info"}


def test_dynamic_config_and_renamed_settings():
    text = MINIMAL.replace(
        "        lprs-postgresql-database:\n",
        """        dynamic-config:
            defaults:
                POSTGRES_DEFAULT_COMMAND_CONTROL:
                    network_timeout_ms: 1500
                FALPRS_PG_TOPOLOGY_SETTINGS:
                    frs-postgresql-database:
                        discovery_timeout_ms: 700
                FALPRS_PG_CACHE_HOST_TYPES:
                    frs-face-descriptor-pg-cache: replica-or-master
                POSTGRES_CONNECTION_POOL_SETTINGS:
                    lprs-postgresql-database:
                        max_pool_size: 20
        lprs-workflow:
            config:
                ban-maintenance-interval: 7s
                pipelines-per-cpu: 8
        frs-workflow:
            config:
                clear-old-events: 2d
                copy-events-maintenance-interval: 1m
        frs-face-descriptor-pg-cache:
            update-interval: 5s
            update-types: only-full
            full-update-op-timeout: 2m
        frs-groups-pg-cache:
            update-interval: 30s
        lprs-postgresql-database:
            max_pool_size: 9
""",
        1,
    )
    config, warnings = convert(text)
    assert warnings == []
    lprs, frs = config["lprs"], config["frs"]
    assert lprs["ban-cleanup-interval"] == "7s"
    assert lprs["pipelines-per-cpu"] == 8
    # The dynamic config's pool size won over the component's
    assert lprs["database"]["max-pool-size"] == 20
    assert lprs["database"]["network-timeout"] == "1500ms"
    assert lprs["database"]["statement-timeout"] == "250ms"
    assert frs["database"]["discovery-timeout"] == "700ms"
    assert frs["events-cleanup-interval"] == "2d"
    assert frs["copy-events-interval"] == "1m"
    # An incremental cache with full updates only
    assert frs["caches"]["face-descriptors"] == {
        "update-interval": "5s",
        "full-update-interval": "0s",
        "host-type": "replica-or-master",
        "full-update-timeout": "2m",
    }
    assert frs["caches"]["groups"] == {"update-interval": "30s"}


def test_changed_dropped_and_unknown_settings_are_warned_about():
    text = (
        "components_manager:\n"
        "    coro_pool:\n"
        "        max_size: 5000\n"
        "    task_processors:\n"
        "        main-task-processor:\n"
        "            worker_threads: 12\n" + MINIMAL.split("components_manager:\n", 1)[1]
    ).replace(
        "        lprs-postgresql-database:\n",
        """        lprs-workflow:
            config:
                screenshot-path: '/data/lprs/'
                failed-ttl: 60
        lprs-postgresql-database:
            min_pool_size: 2
""",
        1,
    )
    config, warnings = convert(text)
    text_of = "\n".join(warnings)
    assert "components_manager.coro_pool.max_size = 5000" in text_of
    assert "lprs-postgresql-database.min_pool_size = 2" in text_of
    assert "lprs-workflow.config.screenshot-path" in text_of
    # worker_threads were set per CPU by the build script: dropped without a warning
    assert "worker_threads" not in text_of
    assert len(warnings) == 3
    # A number without a unit is seconds, as the service parsed it
    assert config["lprs"]["failed-ttl"] == "60s"
    assert "screenshot-path" not in config["lprs"]


def test_invalid_values_are_warned_about_and_left_to_the_defaults():
    text = MINIMAL.replace(
        "        lprs-postgresql-database:\n",
        """        lprs-workflow:
            config:
                events-log-ttl: 'four hours'
                pipelines-per-cpu: 0
        lprs-postgresql-database:
""",
        1,
    )
    config, warnings = convert(text)
    assert len(warnings) == 2
    assert "events-log-ttl" not in config["lprs"]
    assert "pipelines-per-cpu" not in config["lprs"]


def test_command_line(tmp_path):
    destination = tmp_path / "config.toml"
    run = lambda: subprocess.run(  # noqa: E731
        [sys.executable, str(CONVERTER), str(FIXTURES / "config.yaml.example"), str(destination)],
        capture_output=True,
        text=True,
    )
    result = run()
    assert result.returncode == 0, result.stderr
    assert tomllib.loads(destination.read_text())["lprs"]["database"]["max-pool-size"] == 15
    # An existing config is never overwritten
    destination.write_text("# mine\n")
    result = run()
    assert result.returncode == 1
    assert destination.read_text() == "# mine\n"
