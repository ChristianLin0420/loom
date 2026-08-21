from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from loom.train import wandb_util
from scripts import r0_e2e_formal_train_entry as entry


class _Run:
    def __init__(self, *, offline=False):
        self.offline = offline
        self.url = "https://wandb.invalid/run"
        self.defined = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def finish(self):
        self.finished = True


def _environment(monkeypatch, **overrides):
    values = {
        "LOOM_WANDB_PROJECT": "loom-r0-e2e-scratch",
        "LOOM_WANDB_GROUP": "r0-dual-seed0-lineage",
        "LOOM_WANDB_JOB_TYPE": "formal-train",
        "LOOM_WANDB_TAGS": "formal,r0,dual-action",
        "LOOM_WANDB_RESUME": "must",
        "LOOM_WANDB_REQUIRE_ONLINE": "1",
        "WANDB_MODE": "online",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _fake_wandb(monkeypatch, init):
    fake = SimpleNamespace(
        init=init,
        Settings=lambda **kwargs: ("settings", kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_formal_entry_injects_group_job_type_tags_and_link_resume(
    tmp_path, monkeypatch,
):
    _environment(monkeypatch)
    calls = []
    run = _Run()
    fake = _fake_wandb(monkeypatch, lambda **kw: calls.append(kw) or run)
    original = wandb_util.init
    try:
        receipt = entry.install_formal_wandb_contract()
        got = wandb_util.init(
            tmp_path, "loom-r0-e2e-scratch", {"run": {}}, rank=0, name="formal",
        )
    finally:
        wandb_util.init = original
        fake.init = calls.append

    assert got is run
    assert receipt["resume"] == "must"
    assert len(calls) == 1
    assert calls[0]["project"] == "loom-r0-e2e-scratch"
    assert calls[0]["group"] == "r0-dual-seed0-lineage"
    assert calls[0]["job_type"] == "formal-train"
    assert calls[0]["tags"] == ["formal", "r0", "dual-action"]
    assert calls[0]["resume"] == "must"
    assert calls[0]["mode"] == "online"


def test_formal_entry_turns_base_offline_fallback_into_a_hard_failure(
    tmp_path, monkeypatch,
):
    _environment(monkeypatch)
    calls = []

    def init(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise ConnectionError("network down")
        return _Run(offline=True)

    fake = _fake_wandb(monkeypatch, init)
    original = wandb_util.init
    try:
        entry.install_formal_wandb_contract()
        with pytest.raises(entry.FormalWandbError, match="produced no run"):
            wandb_util.init(
                tmp_path, "loom-r0-e2e-scratch", {"run": {}}, rank=0,
            )
    finally:
        wandb_util.init = original
        fake.init = init
    assert len(calls) == 1, "offline retry is rejected before SDK initialization"
    assert os.environ["WANDB_MODE"] == "offline"


def test_formal_entry_fails_closed_on_config_project_mismatch(tmp_path, monkeypatch):
    _environment(monkeypatch)
    calls = []
    fake = _fake_wandb(monkeypatch, lambda **kw: calls.append(kw) or _Run())
    original = wandb_util.init
    try:
        entry.install_formal_wandb_contract()
        with pytest.raises(entry.FormalWandbError, match="produced no run"):
            wandb_util.init(tmp_path, "loom", {"run": {}}, rank=0)
    finally:
        wandb_util.init = original
        fake.init = calls.append
    assert calls == []


@pytest.mark.parametrize("value", ["", "sometimes", "required"])
def test_formal_entry_rejects_invalid_resume_before_training(monkeypatch, value):
    _environment(monkeypatch, LOOM_WANDB_RESUME=value)
    _fake_wandb(monkeypatch, lambda **kw: _Run())
    with pytest.raises(entry.FormalWandbError, match="RESUME"):
        entry.install_formal_wandb_contract()


def test_nonzero_rank_remains_a_noop_under_formal_adapter(tmp_path, monkeypatch):
    _environment(monkeypatch)
    fake = _fake_wandb(monkeypatch, lambda **kw: _Run())
    original = wandb_util.init
    try:
        entry.install_formal_wandb_contract()
        assert wandb_util.init(
            tmp_path, "loom-r0-e2e-scratch", {"run": {}}, rank=7,
        ) is None
    finally:
        wandb_util.init = original
        fake.init = lambda **kw: _Run()
    assert not (tmp_path / "wandb_id").exists()
