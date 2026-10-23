# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from readiness_check import run_vllm_server
from readiness_check.run_vllm_server import (
    ALL_STAGES,
    DEFAULT_STAGES,
    PREFIX_CACHE_THRESHOLDS,
    _attach_serial_reference,
    _counter_delta,
    _parse_prometheus_counters,
    _parse_stages,
    _prefix_cache_verdict,
    _server_command,
)

METRICS = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="org/model"} 4096.0
vllm:prefix_cache_hits_total{engine="0",model_name="org/model"} 2048.0
vllm:external_prefix_cache_queries_total{engine="0",model_name="org/model"} 99.0
vllm:num_requests_running{engine="0",model_name="org/model"} 1.0
"""


def test_new_stages_are_opt_in():
    assert "prefix_cache" in ALL_STAGES
    assert "prefill_burst" in ALL_STAGES
    assert "prefix_cache" not in DEFAULT_STAGES
    assert _parse_stages("serve,prefix_cache,prefill_burst") == ["serve", "prefix_cache", "prefill_burst"]


@pytest.mark.parametrize(
    ("flag", "expected"),
    [(None, []), (True, ["--enable-prefix-caching"]), (False, ["--no-enable-prefix-caching"])],
)
def test_server_command_records_prefix_caching_flag(flag, expected):
    cmd = _server_command(
        hf_model="org/model",
        max_num_seqs=32,
        block_size=64,
        port=8000,
        max_model_len=4096,
        tt_config={"sample_on_device_mode": "all"},
        additional_args=["--async-scheduling"],
        enable_prefix_caching=flag,
    )
    present = [c for c in cmd if "prefix-caching" in c]
    assert present == expected
    assert cmd[-1] == "--async-scheduling"
    assert cmd[cmd.index("--max_model_len") + 1] == "4096"


def test_prometheus_counters_sum_samples_and_fold_total_suffix():
    counters = _parse_prometheus_counters(METRICS)
    assert counters["vllm:prefix_cache_queries"] == 4096.0
    assert counters["vllm:prefix_cache_hits"] == 2048.0
    assert counters["vllm:external_prefix_cache_queries"] == 99.0
    delta = _counter_delta(
        {"vllm:prefix_cache_queries": 4000.0, "vllm:prefix_cache_hits": 2000.0},
        {"vllm:prefix_cache_queries": 4096.0, "vllm:prefix_cache_hits": 2048.0},
    )
    assert delta == {"prefix_cache_queries_delta": 96.0, "prefix_cache_hits_delta": 48.0, "hit_rate": 0.5}


def _request(label, kind, *, ttft, hit_rate, tokens, logprob):
    return {
        "label": label,
        "kind": kind,
        "ttft_ms": ttft,
        "hit_rate": hit_rate,
        "prefix_cache_queries_delta": 2300.0,
        "prefix_cache_hits_delta": 2300.0 * (hit_rate or 0.0),
        "tokens": list(tokens),
        "token_logprobs": [logprob] + [-0.5] * (len(tokens) - 1),
    }


def _record(**overrides):
    tokens = [f"t{i}" for i in range(64)]
    record = {
        "apc_log_enabled": True,
        "requests": [
            _request("cold1", "cold", ttft=1000.0, hit_rate=0.0, tokens=tokens, logprob=-1.0),
            _request("warm1", "warm", ttft=200.0, hit_rate=0.98, tokens=tokens, logprob=-1.1),
            _request("warm2", "warm", ttft=180.0, hit_rate=0.98, tokens=tokens, logprob=-1.05),
        ],
        "shared_prefix_bench": {"hit_rate": 0.82},
    }
    record.update(overrides)
    return record


def test_prefix_cache_verdict_passes_on_hits_fast_ttft_and_identical_output():
    record = _record()
    status, reasons = _prefix_cache_verdict(record, PREFIX_CACHE_THRESHOLDS)
    assert (status, reasons) == ("pass", [])
    assert record["ttft_ratio"] == pytest.approx(0.2)
    assert record["requests"][1]["greedy_identical_through"] == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(apc_log_enabled=False), "prefix caching disabled"),
        (lambda r: r["requests"][1].update(hit_rate=0.2), "warm request hit rate"),
        (lambda r: r["requests"][0].update(hit_rate=0.5), "cold request hit rate"),
        (lambda r: [req.update(ttft_ms=900.0) for req in r["requests"][1:]], "TTFT ratio"),
        (lambda r: r["requests"][1]["tokens"].__setitem__(0, "other"), "first token differs"),
        (lambda r: r["requests"][1]["token_logprobs"].__setitem__(0, -3.0), "logprob delta"),
        (lambda r: r["requests"][2]["tokens"].__setitem__(5, "other"), "diverges after 5 tokens"),
        (lambda r: r.update(shared_prefix_bench={"hit_rate": 0.3}), "shared-prefix benchmark hit rate"),
    ],
)
def test_prefix_cache_verdict_fails_each_gate(mutate, message):
    record = _record()
    mutate(record)
    status, reasons = _prefix_cache_verdict(record, PREFIX_CACHE_THRESHOLDS)
    assert status == "fail"
    assert any(message in reason for reason in reasons), reasons


def test_prefix_cache_verdict_tolerates_bf16_divergence_after_the_hard_window():
    record = _record()
    record["requests"][1]["tokens"][40] = "drift"
    status, reasons = _prefix_cache_verdict(record, PREFIX_CACHE_THRESHOLDS)
    assert status == "pass", reasons
    assert record["requests"][1]["greedy_identical_through"] == 40


def test_attach_serial_reference_records_ttft_ratios(tmp_path):
    serial = tmp_path / "serial.json"
    serial.write_text(json.dumps({"ttft_ms": {"p50": 2000.0, "p99": 5000.0, "mean": 2500.0}}))
    summary_file = tmp_path / "burst.json"
    summary = {"ttft_ms": {"p50": 1000.0, "p99": 2000.0, "mean": 1000.0}}
    result = _attach_serial_reference(summary_file, summary, serial)
    assert result["serial_reference"]["ttft_ratio_p99"] == pytest.approx(0.4)
    assert result["serial_reference"]["ttft_ratio_mean"] == pytest.approx(0.4)
    assert json.loads(summary_file.read_text())["serial_reference"]["ttft_ratio_p50"] == pytest.approx(0.5)
    assert _attach_serial_reference(summary_file, {"ttft_ms": {}}, None) == {"ttft_ms": {}}


def test_module_constants_name_the_counters_the_check_reads():
    assert run_vllm_server.PREFIX_CACHE_COUNTERS == ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits")
