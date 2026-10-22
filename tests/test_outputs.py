"""Scientific acceptance of the published finite-wall vessel contract.

Expected thermodynamics and the three strict-VL histories use an independently
written PR model. The phase-changing closed history shares thermo TP physics
with the reference and has independent PR state/conservation checks. Numeric
assertions remain in the test nodes; table construction only arranges data.
"""

import copy
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import pytest

from candidate_proxy import candidate
from independent_pr import phase, tp

HERE = Path(__file__).parent
DATA = json.loads((HERE / "data/cycles.json").read_text())
DATA["phasewithdrawal"] = json.loads((HERE / "data/phase_withdrawal.json").read_text())
BOUNDARIES = json.loads((HERE / "data/response_boundaries.json").read_text())
STATIC = [(280.0, 1e6, 0.5), (300.0, 1e6, 0.5), (340.0, 1e6, 0.5), (320.0, 2e6, 0.5)]
HISTORIES = ["hysteresis", "selective", "phasecycle", "liquid_history", "phasewithdrawal"]
SINGLE_PHASE = [
    pytest.param(240.0, 1e5, 0.1, 7.3, "l", id="cold_pentane_rich_liquid"),
    pytest.param(260.0, 8e6, 0.5, 2.0, "l", id="compressed_liquid"),
    pytest.param(430.0, 1e5, 0.9, 30.0, "v", id="hot_dilute_vapor"),
    pytest.param(380.0, 1e6, 0.5, 7.3, "v", id="single_vapor"),
]


def run_candidate(payload, timeout=90):
    """Compare the public JSON representation, preserving finite numbers and bools.

    JSON arrays may originate from Python lists or tuples. Their container
    spelling is not part of the public function contract.
    """
    raw = candidate.run(payload, timeout=timeout)
    return json.loads(json.dumps(raw, allow_nan=False))


@lru_cache(maxsize=None)
def cycle(name):
    """Cache one isolated candidate request for each full physical history."""
    payload = copy.deepcopy(DATA[name]["input"])
    timeout = 180 if payload.get("response", False) else 90
    return run_candidate(payload, timeout=timeout)


def composition_table(samples):
    """Arrange optional compositions after their presence is checked separately.

    Zero padding is only a numeric representation of an absent optional field;
    it is never treated as a physical composition or a substitute for nullness.
    """
    return np.asarray(
        [
            (sample["x"] or [0.0, 0.0]) + (sample["y"] or [0.0, 0.0])
            for sample in samples
        ]
    )


def independent_phase_totals(sample):
    """Calculate physical U,V contributions; no candidate/source code imported."""
    energy, volume = 0.0, 0.0
    for amount, composition, kind in (
        (sample["Nv"], sample["y"], "v"),
        (sample["Nl"], sample["x"], "l"),
    ):
        if composition is not None:
            props = phase(sample["T"], sample["P"], composition, kind)
            energy += amount * props["u"]
            volume += amount * props["v"]
    return energy, volume


def invalid_requests():
    """Construct named malformed classes without dispatching in test bodies."""
    thresholds = copy.deepcopy(DATA["selective"]["input"])
    thresholds["controllers"]["l"] = {"off": 1.0, "on": 0.5}
    times = copy.deepcopy(DATA["selective"]["input"])
    times["times"] = [0.0, 10.0, 5.0]
    operation = copy.deepcopy(DATA["selective"]["input"])
    operation["operation"] = "unknown"
    empty = copy.deepcopy(DATA["selective"]["input"])
    empty.update(schedule=[], times=[0.0], response=False)
    boolean = copy.deepcopy(DATA["selective"]["input"])
    boolean["schedule"][0]["enabled"] = ["false", "false"]
    response = copy.deepcopy(DATA["selective"]["input"])
    response["response"] = "false"
    negative = {"operation": "equilibrium", "n": [-1.0, 5.0], "V": 0.01, "U": -10000.0}
    missing = {"operation": "equilibrium", "T": 300.0, "P": 1e6, "z": [0.5, 0.5]}
    nonfinite = {"operation": "equilibrium", "T": float("nan"), "P": 1e6, "z": [0.5, 0.5], "N": 10.0}
    return [
        pytest.param(thresholds, id="thresholds"),
        pytest.param(times, id="times"),
        pytest.param(operation, id="operation"),
        pytest.param(empty, id="empty_schedule"),
        pytest.param(boolean, id="nonboolean_enabled"),
        pytest.param(response, id="nonboolean_response"),
        pytest.param(negative, id="nonpositive_inventory"),
        pytest.param(missing, id="missing_required_field"),
        pytest.param(nonfinite, id="nonfinite_number"),
    ]


@pytest.mark.parametrize("T,P,z", STATIC)
def test_tp_against_independent_pr(T, P, z):
    """“两相存在时，各组分气液逸度相等。” 独立PR与公开焓基准校核TP状态。"""
    expected = tp(T, P, [z, 1 - z])
    actual = run_candidate(
        {"operation": "equilibrium", "T": T, "P": P, "z": [z, 1 - z], "N": 7.3}
    )
    assert actual["T"] == pytest.approx(T, rel=0, abs=0.003)
    assert actual["P"] == pytest.approx(P, rel=3e-5, abs=30)
    assert actual["V"] == pytest.approx(7.3 * expected["v"], rel=5e-5, abs=1e-7)
    assert actual["U"] == pytest.approx(7.3 * expected["u"], rel=5e-5, abs=0.05)
    assert actual["Nv"] == pytest.approx(7.3 * expected["beta"], rel=5e-5, abs=1e-7)
    assert actual["Nl"] == pytest.approx(
        7.3 * (1 - expected["beta"]), rel=5e-5, abs=1e-7
    )
    assert actual["y"] == pytest.approx(expected["gas"]["z"], rel=5e-5, abs=1e-7)
    assert actual["x"] == pytest.approx(expected["liquid"]["z"], rel=5e-5, abs=1e-7)
    assert actual["h_v"] == pytest.approx(expected["gas"]["h"], rel=5e-5, abs=0.05)
    assert actual["h_l"] == pytest.approx(expected["liquid"]["h"], rel=5e-5, abs=0.05)


@pytest.mark.parametrize("T,P,z", STATIC)
def test_extensive_uv_reconstruction(T, P, z):
    """“第二种输入的 `n` 是各组分 mol，`U` 为总 J，不能直接当成原生闪蒸 API 的摩尔输入。”"""
    expected = tp(T, P, [z, 1 - z])
    N = 11.7
    actual = run_candidate(
        {
            "operation": "equilibrium",
            "n": [N * z, N * (1 - z)],
            "V": N * expected["v"],
            "U": N * expected["u"],
        }
    )
    assert actual["T"] == pytest.approx(T, rel=0, abs=0.003)
    assert actual["P"] == pytest.approx(P, rel=3e-5, abs=30)
    assert actual["Nv"] == pytest.approx(N * expected["beta"], rel=5e-5, abs=1e-7)
    assert actual["x"] == pytest.approx(expected["liquid"]["z"], rel=5e-5, abs=1e-7)
    assert actual["y"] == pytest.approx(expected["gas"]["z"], rel=5e-5, abs=1e-7)
    assert actual["V"] == pytest.approx(N * expected["v"], rel=5e-5, abs=1e-7)
    assert actual["U"] == pytest.approx(N * expected["u"], rel=5e-5, abs=0.05)
    assert actual["Nl"] == pytest.approx(N * (1-expected["beta"]), rel=5e-5, abs=1e-7)
    assert actual["h_v"] == pytest.approx(expected["gas"]["h"], rel=5e-5, abs=0.05)
    assert actual["h_l"] == pytest.approx(expected["liquid"]["h"], rel=5e-5, abs=0.05)


@pytest.mark.parametrize("name", HISTORIES)
def test_coupled_trajectory_matches_independent_history(name):
    """“相变不会重置总库存、内能、壁温或累计产品。” 五种完整历史逐字段验收。"""
    actual, expected = cycle(name), DATA[name]["independent"]
    got, want = actual["samples"], expected["samples"]
    assert len(got) == len(want) == len(DATA[name]["input"]["times"]) > 0
    assert actual["volume"] == pytest.approx(expected["volume"], rel=5e-5, abs=1e-7)
    assert actual["initial_energy"] == pytest.approx(
        expected["initial_energy"], rel=5e-5, abs=0.05
    )
    got_time = np.asarray([s["time"] for s in got])
    want_time = np.asarray([s["time"] for s in want])
    assert got_time == pytest.approx(want_time, rel=0, abs=1e-10), (
        f"{name}.sample times"
    )
    got_temperature = np.asarray([[s["T"], s["Tw"]] for s in got])
    want_temperature = np.asarray([[s["T"], s["Tw"]] for s in want])
    assert got_temperature == pytest.approx(want_temperature, rel=0, abs=0.03), (
        f"{name}.samples.[T,Tw]"
    )
    got_pressure = np.asarray([s["P"] for s in got])
    want_pressure = np.asarray([s["P"] for s in want])
    assert got_pressure == pytest.approx(want_pressure, rel=3e-4, abs=600), (
        f"{name}.samples.P"
    )
    got_amounts = np.asarray(
        [s["n"] + [s["Nv"], s["Nl"]] + s["product_v"] + s["product_l"] for s in got]
    )
    want_amounts = np.asarray(
        [s["n"] + [s["Nv"], s["Nl"]] + s["product_v"] + s["product_l"] for s in want]
    )
    assert got_amounts == pytest.approx(want_amounts, rel=3e-4, abs=3e-4), (
        f"{name}.samples.[n0,n1,Nv,Nl,Mv0,Mv1,Ml0,Ml1]"
    )
    got_energies = np.asarray(
        [[s["U"], s["enthalpy_v"], s["enthalpy_l"], s["Q_external"]] for s in got]
    )
    want_energies = np.asarray(
        [[s["U"], s["enthalpy_v"], s["enthalpy_l"], s["Q_external"]] for s in want]
    )
    assert got_energies == pytest.approx(want_energies, rel=5e-4, abs=3), (
        f"{name}.samples.[U,Hv,Hl,Qext]"
    )
    assert [(s["x"] is None, s["y"] is None) for s in got] == [
        (s["x"] is None, s["y"] is None) for s in want
    ], f"{name}.phase presence"
    got_composition, want_composition = composition_table(got), composition_table(want)
    assert got_composition == pytest.approx(want_composition, rel=0, abs=3e-4), (
        f"{name}.samples.[x0,x1,y0,y1]"
    )
    assert [s["on"] for s in got] == [s["on"] for s in want], f"{name}.actual switches"


@pytest.mark.parametrize("name", HISTORIES)
def test_component_and_wall_energy_conservation(name):
    """“采出携带的是摩尔**焓**，罐内储存的是总**内能**。” 所有请求点同时满足组分与含壁能量收支。"""
    actual = cycle(name)
    samples = actual["samples"]
    initial, wall = DATA[name]["input"]["initial"], DATA[name]["input"]["wall"]
    assert len(samples) == len(DATA[name]["input"]["times"]) > 0
    inventory = np.array(initial["z"]) * initial["N"]
    got_total = np.asarray(
        [
            np.asarray(s["n"]) + s["product_v"] + np.asarray(s["product_l"])
            for s in samples
        ]
    )
    want_total = np.tile(inventory, (len(samples), 1))
    assert got_total == pytest.approx(want_total, rel=0, abs=3e-5 * initial["N"]), (
        f"{name}.component inventory + both products"
    )
    got_balance = np.asarray(
        [
            s["U"]
            + wall["C"] * (s["Tw"] - initial["Tw"])
            + s["enthalpy_v"]
            + s["enthalpy_l"]
            - s["Q_external"]
            for s in samples
        ]
    )
    want_balance = np.full(len(samples), actual["initial_energy"])
    assert got_balance == pytest.approx(
        want_balance, rel=0, abs=max(2, 3e-5 * abs(actual["initial_energy"]))
    ), f"{name}.fluid + wall + effluent - external heat"
    phase_moles = np.asarray([s["Nv"] + s["Nl"] for s in samples])
    inventory_moles = np.asarray([sum(s["n"]) for s in samples])
    independent_samples = DATA[name]["independent"]["samples"]
    phase_total_budget = np.asarray([
        sum(max(3e-4, 3e-4*abs(value)) for value in [s["Nv"], s["Nl"]]+s["n"])
        for s in independent_samples
    ])
    assert (np.abs(phase_moles-inventory_moles) <= phase_total_budget).all(), f"{name}.phase/total moles"


@pytest.mark.parametrize("name", ["hysteresis", "selective", "liquid_history", "phasewithdrawal"])
def test_hysteretic_and_command_event_history(name):
    """“上下限之间保持原状态。” 检验气/液路滞回与工况命令的事件身份和时间。"""
    actual, expected = cycle(name)["events"], DATA[name]["independent"]["events"]
    assert len(actual) == len(expected) > 0
    got = [(e["phase"], e["kind"], e["before"], e["after"], e["cause"]) for e in actual]
    want = [
        (e["phase"], e["kind"], e["before"], e["after"], e["cause"]) for e in expected
    ]
    assert got == want, f"{name}.events.[phase,kind,before,after,cause]"
    got_time, want_time = (
        np.asarray([e["time"] for e in actual]),
        np.asarray([e["time"] for e in expected]),
    )
    assert got_time == pytest.approx(want_time, rel=0, abs=0.03), f"{name}.event times"


def test_phase_appearance_disappearance_and_static_caloric_consistency():
    """“不存在的相库存为零，组成和摩尔焓为 `null`，不能沿用消失前的组成。” 独立PR验证闭罐相变状态。"""
    result = cycle("phasecycle")
    samples = result["samples"]
    assert len(samples) == len(DATA["phasecycle"]["input"]["times"]) > 3
    assert samples[0]["Nl"] > 0.1
    assert samples[3]["Nl"] == pytest.approx(0, abs=1e-7)
    assert samples[3]["x"] is None
    assert samples[-1]["Nl"] > 0.1
    # Reconstruct the physical expectation at the independently checked stored
    # state. Candidate state fields each have their own public error budget;
    # reapplying a nonlinear EOS to those approximations would mix those errors.
    independent = np.asarray([independent_phase_totals(s) for s in DATA["phasecycle"]["independent"]["samples"]])
    got_energy = np.asarray([s["U"] for s in samples])
    assert got_energy == pytest.approx(independent[:, 0], rel=5e-4, abs=3), (
        "phasecycle.U independently reconstructed from PR phases"
    )
    got_volume = np.full(len(samples), result["volume"])
    assert got_volume == pytest.approx(independent[:, 1], rel=5e-4, abs=1e-7), (
        "phasecycle.V independently reconstructed from PR phases"
    )
    inventories = np.asarray([sum(s["n"]) for s in samples])
    assert (inventories > 0).all(), "phasecycle positive component inventory"


@pytest.mark.parametrize("name", ["hysteresis", "selective", "phasewithdrawal"])
def test_parameter_total_response_includes_moving_events(name):
    """“参数邻域内相平衡、物料、能量和事件位置均随参数改变。” 整周期导数包括事件移动。"""
    got, want = cycle(name)["response"], DATA[name]["independent"]["response"]
    assert isinstance(got, dict)
    got_products = np.asarray(got["product_v"] + got["product_l"])
    want_products = np.asarray(want["product_v"] + want["product_l"])
    assert got_products == pytest.approx(want_products, rel=0.03, abs=0.0003), (
        f"{name}.d[Mv0,Mv1,Ml0,Ml1]/dtheta"
    )
    assert got["event_times"] == pytest.approx(
        want["event_times"], rel=0.03, abs=0.05
    ), f"{name}.d(event time)/dtheta"


def test_independent_calls_and_input_immutability():
    """“不同请求的状态须独立，包括上一请求的相态、开关和累计产品。”"""
    p = {"operation": "equilibrium", "T": 300.0, "P": 1e6, "z": [0.35, 0.65], "N": 8.0}
    before = copy.deepcopy(p)
    first = run_candidate(p)
    run_candidate(
        {"operation": "equilibrium", "T": 380.0, "P": 1e6, "z": [0.7, 0.3], "N": 3.0}
    )
    again = run_candidate(p)
    assert p == before
    expected = tp(p["T"], p["P"], p["z"])
    for got in (first, again):
        assert got["T"] == pytest.approx(p["T"], rel=0, abs=0.003)
        assert got["P"] == pytest.approx(p["P"], rel=3e-5, abs=30)
        assert got["U"] == pytest.approx(p["N"] * expected["u"], rel=5e-5, abs=0.05)
        assert got["V"] == pytest.approx(p["N"] * expected["v"], rel=5e-5, abs=1e-7)
        assert got["Nv"] == pytest.approx(p["N"] * expected["beta"], rel=5e-5, abs=1e-7)
        assert got["Nl"] == pytest.approx(p["N"] * (1-expected["beta"]), rel=5e-5, abs=1e-7)


def test_terminal_inventory_event_is_right_continuous():
    """“请求时刻恰等于事件时，输出右连续的开停状态；其他连续量不跳变。” 单气相N=10-0.1t解析关闭。"""
    p = {
        "operation": "simulate",
        "initial": {"T": 380.0, "P": 1e6, "z": [0.5, 0.5], "N": 10.0, "Tw": 380.0},
        "wall": {"C": 600.0, "Ki": 20.0, "Ko": 0.0},
        "controllers": {"v": {"off": 8.0, "on": 9.0}, "l": {"off": 0.2, "on": 0.5}},
        "schedule": [
            {
                "until": 20.0,
                "ambient": 380.0,
                "flow": [0.1, 0.0],
                "enabled": [True, False],
            }
        ],
        "times": [0.0, 20.0],
        "response": False,
    }
    got = run_candidate(p)
    final = got["samples"][-1]
    assert final["Nv"] == pytest.approx(8.0, rel=0, abs=3e-4)
    assert final["n"] == pytest.approx([4.0, 4.0], rel=0, abs=3e-4)
    assert final["Nl"] == pytest.approx(0.0, abs=1e-7)
    assert final["on"] == [False, False]
    assert len(got["events"]) == 2
    assert got["events"][-1]["phase"] == "v"
    assert got["events"][-1]["kind"] == "close"
    assert got["events"][-1]["cause"] == "inventory"
    assert got["events"][-1]["time"] == pytest.approx(20.0, rel=0, abs=0.03)


@pytest.mark.parametrize("payload", invalid_requests())
def test_public_invalid_input_errors(payload):
    """“非法结构、非正库存、无效阈值、非法时间序列或未知 operation 抛 `ValueError`” 九类真实无效输入。"""
    with pytest.raises(ValueError):
        run_candidate(payload)


@pytest.mark.parametrize(
    "boundary_id", ["near_terminal", "near_command", "small_theta"]
)
def test_response_local_branch_and_scale_match_independent_derivative(boundary_id):
    """“但不能用此拒绝支持域内的正常响应请求。” 合法窄事件邻域与小theta均应返回完整导数。"""
    case = BOUNDARIES[boundary_id]
    got = run_candidate(copy.deepcopy(case["input"]), timeout=180)
    expected = case["independent"]
    assert len(got["events"]) == len(expected["events"]) > 0
    actual_signature = [(e["phase"], e["kind"], e["cause"]) for e in got["events"]]
    expected_signature = [
        (e["phase"], e["kind"], e["cause"]) for e in expected["events"]
    ]
    assert actual_signature == expected_signature, f"{boundary_id}.local event branch"
    actual_times = np.asarray([e["time"] for e in got["events"]])
    expected_times = np.asarray([e["time"] for e in expected["events"]])
    assert actual_times == pytest.approx(expected_times, rel=0, abs=0.03), (
        f"{boundary_id}.event locations"
    )
    response, wanted = got["response"], case["response"]
    assert isinstance(response, dict), f"{boundary_id}.response must be defined"
    got_products = np.asarray(response["product_v"] + response["product_l"])
    want_products = np.asarray(wanted["product_v"] + wanted["product_l"])
    assert got_products == pytest.approx(want_products, rel=0.03, abs=0.0003), (
        f"{boundary_id}.total product derivatives"
    )
    assert response["event_times"] == pytest.approx(
        wanted["event_times"], rel=0.03, abs=0.05
    ), f"{boundary_id}.event-time derivatives"


@pytest.mark.parametrize("T,P,z,N,kind", SINGLE_PHASE)
@pytest.mark.parametrize("specification", ["TP", "UV"])
def test_stable_single_phase_equilibrium_full_state(T, P, z, N, kind, specification):
    """“单相也必须对另一相稳定。” 独立PR纯相根覆盖温压/库存边界及总量UV的完整返回。"""
    composition = [z, 1-z]
    expected = phase(T, P, composition, kind)
    payloads = {
        "TP": {"operation": "equilibrium", "T": T, "P": P, "z": composition, "N": N},
        "UV": {"operation": "equilibrium", "n": [N*z, N*(1-z)], "V": N*expected["v"], "U": N*expected["u"]},
    }
    got = run_candidate(payloads[specification])
    present, absent = {"v": (("Nv", "y", "h_v"), ("Nl", "x", "h_l")),
                       "l": (("Nl", "x", "h_l"), ("Nv", "y", "h_v"))}[kind]
    assert got["T"] == pytest.approx(T, rel=0, abs=0.003)
    assert got["P"] == pytest.approx(P, rel=3e-5, abs=30)
    assert got["V"] == pytest.approx(N*expected["v"], rel=5e-5, abs=1e-7)
    assert got["U"] == pytest.approx(N*expected["u"], rel=5e-5, abs=0.05)
    assert got[present[0]] == pytest.approx(N, rel=5e-5, abs=1e-7)
    assert got[present[1]] == pytest.approx(composition, rel=5e-5, abs=1e-7)
    assert got[present[2]] == pytest.approx(expected["h"], rel=5e-5, abs=0.05)
    assert got[absent[0]] == pytest.approx(0.0, rel=0, abs=1e-7)
    assert got[absent[1]] is None
    assert got[absent[2]] is None
    assert set(got) >= {"T", "P", "V", "U", "Nv", "Nl", "y", "x", "h_v", "h_l"}
    assert isinstance(got, dict)
    scalars = [got[key] for key in ("T", "P", "V", "U", "Nv", "Nl", "h_v", "h_l") if got[key] is not None]
    compositions = [got[key] for key in ("x", "y") if got[key] is not None]
    assert all(len(value) == 2 for value in compositions)
    numbers = scalars + [value for composition in compositions for value in composition]
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers)


@pytest.mark.parametrize("name", HISTORIES)
def test_simulation_public_schema_and_finite_json(name):
    """“返回普通JSON兼容的Python数值和容器，不要求字典键顺序或字节级 JSON 格式。” 仅检验公开结构与类型。"""
    result = cycle(name)
    assert set(result) >= {"volume", "initial_energy", "samples", "events", "response"}
    assert isinstance(result, dict)
    sample_keys = {"time", "T", "P", "Tw", "n", "U", "Nv", "Nl", "y", "x", "on", "product_v", "product_l", "enthalpy_v", "enthalpy_l", "Q_external"}
    assert [set(s) >= sample_keys for s in result["samples"]] == [True]*len(result["samples"])
    vector_lengths = [[len(s[key]) for key in ("n", "product_v", "product_l", "on")] for s in result["samples"]]
    assert vector_lengths == [[2, 2, 2, 2]]*len(result["samples"])
    switches = [value for sample in result["samples"] for value in sample["on"]]
    assert all(type(value) is bool for value in switches)
    event_switches = [value for event in result["events"] for value in (event["before"], event["after"])]
    assert all(type(value) is bool for value in event_switches)
    assert [set(e) >= {"time", "phase", "kind", "before", "after", "cause"} for e in result["events"]] == [True]*len(result["events"])
    scalar_fields = ("time", "T", "P", "Tw", "U", "Nv", "Nl", "enthalpy_v", "enthalpy_l", "Q_external")
    scalars = [s[key] for s in result["samples"] for key in scalar_fields]
    compositions = [s[key] for s in result["samples"] for key in ("x", "y") if s[key] is not None]
    assert all(len(value) == 2 for value in compositions)
    numbers = [result["volume"], result["initial_energy"]] + scalars
    numbers += [v for s in result["samples"] for key in ("n", "product_v", "product_l") for v in s[key]]
    numbers += [v for composition in compositions for v in composition]
    numbers += [event["time"] for event in result["events"]]
    if result["response"] is not None:
        response = result["response"]
        assert isinstance(response, dict)
        assert set(response) >= {"product_v", "product_l", "event_times"}
        assert [len(response[key]) for key in ("product_v", "product_l", "event_times")] == [2, 2, len(result["events"])]
        numbers += [v for key in ("product_v", "product_l", "event_times") for v in response[key]]
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers)


@pytest.mark.parametrize("name", ["phasecycle", "liquid_history"])
def test_disabled_response_returns_null(name):
    """“当 `response=false` 或省略时，返回 `response:null`。” 两个不求响应的完整周期。"""
    assert cycle(name)["response"] is None


def test_phase_change_without_control_does_not_create_switch_events():
    """“纯相出现/消失不单独增加开停事件。” 闭罐VL→V→VL且两路始终命令禁用。"""
    result = cycle("phasecycle")
    assert result["events"] == []
    assert [s["on"] for s in result["samples"]] == [[False, False]]*len(result["samples"])


def command_only_payload():
    """A constant thermodynamic state with only externally scheduled switches."""
    return {
        "operation": "simulate",
        "initial": {"T": 300.0, "P": 1e6, "z": [0.5, 0.5], "N": 10.0, "Tw": 300.0},
        "wall": {"C": 600.0, "Ki": 20.0, "Ko": 0.0},
        "controllers": {"v": {"off": 0.2, "on": 0.5}, "l": {"off": 0.3, "on": 0.7}},
        "schedule": [
            {"until": 1.0, "ambient": 300.0, "flow": [0.0, 0.0], "enabled": [False, True]},
            {"until": 2.0, "ambient": 300.0, "flow": [0.0, 0.0], "enabled": [True, False]},
            {"until": 3.0, "ambient": 300.0, "flow": [0.0, 0.0], "enabled": [True, True]},
        ],
        "times": [0.0, 1.0, 2.0, 3.0],
        "response": True,
    }


def test_fixed_simultaneous_commands_zero_flow_and_response():
    """“固定命令本身的同时开停允许，按上一节的既有顺序处理，其时刻导数为0” 等温零流的解析完整响应。"""
    got = run_candidate(command_only_payload(), timeout=180)
    assert [(e["phase"], e["kind"], e["cause"]) for e in got["events"]] == [
        ("l", "open", "command"), ("l", "close", "command"),
        ("v", "open", "command"), ("l", "open", "command"),
    ]
    assert [e["time"] for e in got["events"]] == pytest.approx([0.0, 1.0, 1.0, 2.0], rel=0, abs=0.03)
    assert [s["on"] for s in got["samples"]] == [[False, True], [True, False], [True, True], [True, True]]
    assert np.asarray([s["n"] for s in got["samples"]]) == pytest.approx(np.full((4, 2), 5.0), rel=3e-4, abs=3e-4)
    assert np.asarray([[s["T"], s["Tw"]] for s in got["samples"]]) == pytest.approx(np.full((4, 2), 300.0), rel=0, abs=0.03)
    assert np.asarray([s["product_v"]+s["product_l"] for s in got["samples"]]) == pytest.approx(np.zeros((4, 4)), rel=0, abs=3e-4)
    assert got["response"]["product_v"] == pytest.approx([0.0, 0.0], rel=0, abs=0.0003)
    assert got["response"]["product_l"] == pytest.approx([0.0, 0.0], rel=0, abs=0.0003)
    assert got["response"]["event_times"] == pytest.approx([0.0]*4, rel=0, abs=0.05)


def test_dynamic_requests_reset_phase_switches_and_products():
    """“不同请求的状态须独立，包括上一请求的相态、开关和累计产品。” 两相采出→纯气闭罐→同一两相请求。"""
    p = copy.deepcopy(DATA["selective"]["input"])
    p.update(schedule=[dict(p["schedule"][0], until=4.0)], times=(0.0, 2.0, 4.0))
    p.pop("response")
    before = copy.deepcopy(p)
    first = run_candidate(p)
    middle = command_only_payload()
    middle["initial"].update(T=380.0, Tw=380.0, N=3.0)
    middle.update(response=False, schedule=[dict(middle["schedule"][0], until=1.0, enabled=[False, False])], times=[0.0, 1.0])
    run_candidate(middle)
    again = run_candidate(p)
    assert p == before
    assert first["response"] is None and again["response"] is None
    assert [(e["phase"], e["kind"], e["cause"], e["before"], e["after"]) for e in again["events"]] == [
        (e["phase"], e["kind"], e["cause"], e["before"], e["after"]) for e in first["events"]
    ]
    assert [e["time"] for e in again["events"]] == pytest.approx([e["time"] for e in first["events"]], rel=0, abs=2*0.03)
    assert again["samples"][0]["product_v"] == [0.0, 0.0]
    assert again["samples"][0]["product_l"] == [0.0, 0.0]
    assert again["samples"][0]["on"] == [True, True]
    assert again["samples"][0]["Tw"] == pytest.approx(300.0, rel=0, abs=0.03)
    amounts = ("n", "product_v", "product_l")
    got = np.asarray([s[k] for s in again["samples"] for k in amounts])
    want = np.asarray([s[k] for s in first["samples"] for k in amounts])
    repeat_budget = np.maximum(3e-4, 3e-4*np.abs(got)) + np.maximum(3e-4, 3e-4*np.abs(want))
    assert (np.abs(got-want) <= repeat_budget).all()
    assert [s["x"] is None for s in again["samples"]] == [False]*3


def test_phase_transition_keeps_selective_products_and_moving_event_response():
    """“相变不会重置总库存、内能、壁温或累计产品。” 独立PR/RK45的双路采出VL→V→VL与液路关闭/重开。"""
    result = cycle("phasewithdrawal")
    samples = result["samples"]
    assert samples[0]["Nl"] > 0.1
    assert samples[3]["Nl"] == pytest.approx(0.0, rel=0, abs=1e-7)
    assert samples[3]["x"] is None
    assert samples[-1]["Nl"] > 0.1
    assert samples[3]["on"] == [True, False]
    assert samples[-1]["on"] == [True, True]
    expected = DATA["phasewithdrawal"]["independent"]
    plateau_budget = [
        max(3e-4, 3e-4*abs(a)) + max(3e-4, 3e-4*abs(b))
        for a,b in zip(expected["samples"][3]["product_l"], expected["samples"][2]["product_l"])
    ]
    assert (np.abs(np.asarray(samples[3]["product_l"])-samples[2]["product_l"]) <= plateau_budget).all()
    assert sum(samples[-1]["product_l"]) > sum(samples[3]["product_l"]) + 0.01
    gas_product_budget = sum(max(3e-4, 3e-4*abs(value)) for value in expected["samples"][-1]["product_v"])
    assert sum(samples[-1]["product_v"]) == pytest.approx(0.005*180.0, rel=0, abs=gas_product_budget)
    derivatives = result["response"]
    expected_derivatives = expected["response"]
    gas_derivative_budget = sum(max(0.0003, 0.03*abs(value)) for value in expected_derivatives["product_v"])
    assert sum(derivatives["product_v"]) == pytest.approx(0.0, rel=0, abs=gas_derivative_budget)
    # For constant liquid flow, total product is Fl*(t_close + tf-t_open).
    # Its derivative gives a separate analytic check of the moving intervals.
    liquid_interval_derivative = 0.004*(derivatives["event_times"][2]-derivatives["event_times"][3])
    liquid_identity_budget = sum(max(0.0003, 0.03*abs(value)) for value in expected_derivatives["product_l"])
    liquid_identity_budget += 0.004*sum(max(0.05, 0.03*abs(expected_derivatives["event_times"][index])) for index in (2,3))
    assert sum(derivatives["product_l"]) == pytest.approx(liquid_interval_derivative, rel=0, abs=liquid_identity_budget)
