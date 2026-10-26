"""Merge regression checks: fake credentials, fake dotenv and no DB connection."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def connection_module(monkeypatch):
    suffixes = ("HOST", "PORT", "DB", "USER", "PASSWORD")
    for key in ["PG_" + value for value in suffixes] + [
        "PUBG_DB_" + value for value in ("HOST", "PORT", "NAME", "USER", "PASSWORD")
    ]:
        monkeypatch.delenv(key, raising=False)
    loaded = []
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: loaded.append(True)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    source = Path(__file__).resolve().parents[1] / 'archive/legacy_snapshot/main.py'
    spec = importlib.util.spec_from_file_location("legacy_merge_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        module.psycopg2, "connect", lambda **config: calls.append(config) or sentinel
    )
    return module, loaded, calls, sentinel


def test_import_does_not_load_private_settings(connection_module):
    _, loaded, calls, _ = connection_module
    assert not loaded
    assert not calls


@pytest.mark.parametrize("prefix", ["PG_", "PUBG_DB_"])
def test_both_existing_environment_contracts(connection_module, monkeypatch, prefix):
    module, loaded, calls, sentinel = connection_module
    original = dict(module.DB_CONFIG)
    expected = dict(host="fixture-host", port="5433", dbname="fixture-db",
                    user="fixture-user", password="fixture-password")
    for field, value in expected.items():
        suffix = ("DB" if prefix == "PG_" else "NAME") if field == "dbname" else field.upper()
        monkeypatch.setenv(prefix + suffix, value)
    assert module.get_conn() is sentinel
    assert len(loaded) == 1
    assert all(calls[0][key] == value for key, value in expected.items())
    assert module.DB_CONFIG == original


def test_pg_variables_take_precedence(connection_module, monkeypatch):
    module, _, calls, _ = connection_module
    for prefix, value in [("PG_", "primary-fixture"), ("PUBG_DB_", "alias-fixture")]:
        monkeypatch.setenv(prefix + "PASSWORD", value)
        monkeypatch.setenv(prefix + "HOST", value)
    module.get_conn()
    assert calls[0]["host"] == "primary-fixture"
    assert calls[0]["password"] == "primary-fixture"


def test_empty_primary_password_uses_alias(connection_module, monkeypatch):
    module, _, calls, _ = connection_module
    monkeypatch.setenv("PG_PASSWORD", "")
    monkeypatch.setenv("PUBG_DB_PASSWORD", "alias-fixture")
    module.get_conn()
    assert calls[0]["password"] == "alias-fixture"


def test_missing_password_fails_before_connect(connection_module):
    module, _, calls, _ = connection_module
    with pytest.raises(RuntimeError, match="PG_PASSWORD"):
        module.get_conn()
    assert not calls
