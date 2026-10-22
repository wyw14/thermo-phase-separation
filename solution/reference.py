"""Author reference: adaptive ODE with native stable UV flashes and events."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root, least_squares
from thermo_model import make_flasher


def _equilibrium(flasher, n, volume, energy, hot=None):
    total = float(sum(n))
    z = [float(x / total) for x in n]
    target_v, target_u = volume / total, energy / total
    guess = [hot.T / 300.0, np.log(hot.P)] if hot is not None else [1.0, np.log(1e6)]
    last = [None]
    lower = np.array([0.75, np.log(6e4)])
    upper = np.array([1.5, np.log(1e7)])

    def residual(v):
        # An unconstrained Newton step is not a physical state. Keep its
        # failures inside this reconstruction and retry the bounded problem.
        if not np.isfinite(v).all() or np.any(v < lower) or np.any(v > upper):
            raise ArithmeticError("TP reconstruction trial left the search box")
        eq = flasher.flash(T=float(300 * v[0]), P=float(np.exp(v[1])), zs=z)
        last[0] = eq
        return [np.log(eq.V() / target_v), (eq.U() - target_u) / 20000.0]

    r = None
    try:
        fitted = root(residual, guess, tol=2e-10)
        if np.linalg.norm(residual(fitted.x), ord=np.inf) <= 2e-8:
            r = fitted
    except (ArithmeticError, ValueError, TypeError):
        pass
    if r is None:
        # Stable single-liquid states can be poorly conditioned from a vapor
        # starting estimate. The public box is contained strictly inside these
        # search bounds; alternate seeds do not change the supported branch.
        seeds = (guess, [0.8, np.log(1e5)], [1.0, np.log(6e6)], [1.4, np.log(1e6)])
        for seed in seeds:
            try:
                fitted = least_squares(
                    residual, np.clip(seed, lower + 1e-12, upper - 1e-12),
                    bounds=(lower, upper), xtol=1e-12, ftol=1e-12,
                    gtol=1e-12, max_nfev=150,
                )
                if np.linalg.norm(residual(fitted.x), ord=np.inf) <= 2e-8:
                    r = fitted
                    break
            except (ArithmeticError, ValueError, TypeError):
                continue
    if r is None or np.linalg.norm(residual(r.x), ord=np.inf) > 2e-7:
        raise RuntimeError("stable VU constraint reconstruction failed")
    return last[0]


def _phase(eq, total):
    if eq.gas is None:
        nv, y, hv = 0.0, None, None
    else:
        nv = total * eq.betas[eq.phases.index(eq.gas)]
        y, hv = list(eq.gas.zs), float(eq.gas.H())
    if not eq.liquids:
        nl, x, hl = 0.0, None, None
    else:
        liquid = eq.liquids[0]
        nl = total * eq.betas[eq.phases.index(liquid)]
        x, hl = list(liquid.zs), float(liquid.H())
    return nv, nl, y, x, hv, hl


def _static(payload):
    f = make_flasher()
    if "n" in payload:
        n = np.asarray(payload["n"], dtype=float)
        eq = _equilibrium(f, n, float(payload["V"]), float(payload["U"]))
        total = float(n.sum())
    else:
        total = float(payload["N"])
        eq = f.flash(T=float(payload["T"]), P=float(payload["P"]), zs=payload["z"])
    nv, nl, y, x, hv, hl = _phase(eq, total)
    return {
        "T": float(eq.T),
        "P": float(eq.P),
        "V": total * eq.V(),
        "U": total * eq.U(),
        "Nv": nv,
        "Nl": nl,
        "y": y,
        "x": x,
        "h_v": hv,
        "h_l": hl,
    }


def _validate_content(p):
    if not isinstance(p, dict):
        raise ValueError("payload must be an object")
    if p.get("operation") not in ("equilibrium", "simulate"):
        raise ValueError("unknown operation")
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a finite real number")
        if not np.isfinite(value):
            raise ValueError("nonfinite input")
        return value

    def pair(values):
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError("expected a two-component array")
        return [number(value) for value in values]

    def tp_spec(spec):
        z = pair(spec["z"])
        if min(z) <= 0 or abs(sum(z) - 1) > 1e-10 or number(spec["N"]) <= 0:
            raise ValueError("invalid TP specification")
        number(spec["T"])
        number(spec["P"])

    if p["operation"] == "equilibrium":
        if "n" in p:
            if min(pair(p["n"])) <= 0 or number(p["V"]) <= 0:
                raise ValueError("invalid UV specification")
            number(p["U"])
        else:
            tp_spec(p)
        return
    initial, wall = p["initial"], p["wall"]
    tp_spec(initial)
    number(initial["Tw"])
    if number(wall["C"]) <= 0 or number(wall["Ki"]) <= 0 or number(wall["Ko"]) < 0 or number(p.get("theta", 1)) <= 0:
        raise ValueError("invalid wall")
    if not isinstance(p.get("response", False), bool):
        raise ValueError("response must be boolean")
    for phase in ("v", "l"):
        ctl = p["controllers"][phase]
        if not (0 < number(ctl["off"]) < number(ctl["on"])):
            raise ValueError("invalid hysteresis thresholds")
    until = 0.0
    if not isinstance(p["schedule"], (list, tuple)) or not p["schedule"]:
        raise ValueError("schedule must contain a segment")
    for segment in p["schedule"]:
        if (
            number(segment["until"]) <= until
            or min(pair(segment["flow"])) < 0
            or len(segment["enabled"]) != 2
            or any(type(value) is not bool for value in segment["enabled"])
        ):
            raise ValueError("invalid schedule")
        number(segment["ambient"])
        until = segment["until"]
    times = p["times"]
    if not isinstance(times, (list, tuple)) or any(number(value) < 0 or value > until for value in times) or list(times) != sorted(set(times)):
        raise ValueError("invalid sample times")
    if p.get("response", False) and (not times or times[-1] != until):
        raise ValueError("response requires a final-time sample")


def _validate(p):
    def finite(value):
        if isinstance(value, dict):
            for element in value.values():
                finite(element)
        elif isinstance(value, (list, tuple)):
            for element in value:
                finite(element)
        elif isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError("nonfinite input")

    try:
        finite(p)
        _validate_content(p)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("invalid request structure") from error


def _cycle(p, theta, *, rtol=1e-7, max_step=1.0):
    initial, wall = p["initial"], p["wall"]
    f = make_flasher()
    eq0 = f.flash(T=initial["T"], P=initial["P"], zs=initial["z"])
    total0 = initial["N"]
    volume = total0 * eq0.V()
    energy0 = total0 * eq0.U()
    # Extensive energies are scaled only inside the integration state.
    scale = 10000.0
    state = np.zeros(11)
    state[:2] = np.asarray(initial["z"]) * total0
    state[2], state[3] = energy0 / scale, initial["Tw"]
    on = [False, False]
    off = [p["controllers"][x]["off"] for x in ("v", "l")]
    high = [p["controllers"][x]["on"] for x in ("v", "l")]
    events, pieces = [], []
    cache = {"key": None, "eq": eq0}

    def flash(y):
        key = tuple(y[:3])
        if key != cache["key"]:
            cache["eq"] = _equilibrium(f, y[:2], volume, y[2] * scale, cache["eq"])
            cache["key"] = key
        return cache["eq"]

    def set_control(t, y, enabled, reason):
        eq = flash(y)
        amounts = _phase(eq, float(sum(y[:2])))[:2]
        before = list(on)
        for j in range(2):
            if not enabled[j]:
                on[j] = False
            elif on[j] and amounts[j] <= off[j] + 1e-8:
                on[j] = False
            elif not on[j] and amounts[j] >= high[j] - 1e-8:
                on[j] = True
        changed = []
        for j in range(2):
            if before[j] != on[j]:
                changed.append(
                    {
                        "time": float(t),
                        "phase": "v" if j == 0 else "l",
                        "kind": "open" if on[j] else "close",
                        "before": before[j],
                        "after": on[j],
                        "cause": reason,
                    }
                )
        events.extend(
            sorted(changed, key=lambda item: (item["after"], item["phase"] != "v"))
        )

    current = 0.0
    for segment in p["schedule"]:
        end = float(segment["until"])
        enabled = segment["enabled"]
        set_control(current, state, enabled, "command")
        while current < end - 1e-10:
            mode = tuple(on)

            def rhs(t, y):
                eq = flash(y)
                _, _, yp, xp, hv, hl = _phase(eq, float(sum(y[:2])))
                fv = segment["flow"][0] if mode[0] and yp is not None else 0.0
                fl = segment["flow"][1] if mode[1] and xp is not None else 0.0
                av = np.asarray(yp if yp is not None else [0.0, 0.0]) * fv
                al = np.asarray(xp if xp is not None else [0.0, 0.0]) * fl
                ev, el = fv * (hv or 0.0), fl * (hl or 0.0)
                qi = theta * wall["Ki"] * (y[3] - eq.T)
                qo = wall["Ko"] * (segment["ambient"] - y[3])
                dy = np.zeros(11)
                dy[:2] = -av - al
                dy[2] = (qi - ev - el) / scale
                dy[3] = (qo - qi) / wall["C"]
                dy[4:6], dy[6:8] = av, al
                dy[8], dy[9], dy[10] = ev / scale, el / scale, qo / scale
                return dy

            watchers = []
            for j in range(2):
                if not enabled[j]:
                    continue
                target = off[j] if mode[j] else high[j]

                def watcher(t, y, index=j, threshold=target):
                    eq = flash(y)
                    return _phase(eq, float(sum(y[:2])))[index] - threshold

                watcher.terminal = True
                watcher.direction = -1.0 if mode[j] else 1.0
                watchers.append(watcher)
            result = solve_ivp(
                rhs,
                (current, end),
                state,
                method="DOP853",
                rtol=rtol,
                atol=np.array([1e-9] * 3 + [1e-7] + [1e-9] * 7),
                max_step=max_step,
                dense_output=True,
                events=watchers,
            )
            if not result.success:
                raise RuntimeError(result.message)
            next_time = float(result.t[-1])
            pieces.append((current, next_time, result.sol, mode))
            state = result.y[:, -1]
            if next_time <= current + 1e-11:
                raise RuntimeError("nonadvancing control event")
            current = next_time
            if current < end - 1e-10:
                set_control(current, state, enabled, "inventory")
                if tuple(on) == mode:
                    raise RuntimeError("unresolved control event")
            if len(events) > 64:
                raise RuntimeError("unsupported event count")
    # At the terminal boundary there is no next schedule iteration to perform
    # the right-continuous update. A threshold reached exactly there still
    # changes the physical switch and belongs in the event history.
    set_control(current, state, p["schedule"][-1]["enabled"], "inventory")
    # A request at final time uses the final active schedule and right-continuous mode.
    samples = []
    for t in p["times"]:
        matching = [
            piece for piece in pieces if piece[0] - 1e-9 <= t <= piece[1] + 1e-9
        ]
        if not matching:
            y, mode = state, tuple(on)
        else:
            a, b, interpolation, mode = matching[-1]
            y = interpolation(min(max(float(t), a), b))
        right_events = [event for event in events if abs(event["time"] - t) <= 1e-8]
        mode = list(mode)
        for event in right_events:
            mode[0 if event["phase"] == "v" else 1] = event["after"]
        eq = flash(y)
        nv, nl, yp, xp, _, _ = _phase(eq, float(sum(y[:2])))
        samples.append(
            {
                "time": float(t),
                "T": float(eq.T),
                "P": float(eq.P),
                "Tw": float(y[3]),
                "n": y[:2].tolist(),
                "U": float(y[2] * scale),
                "Nv": nv,
                "Nl": nl,
                "y": yp,
                "x": xp,
                "on": list(mode),
                "product_v": y[4:6].tolist(),
                "product_l": y[6:8].tolist(),
                "enthalpy_v": float(y[8] * scale),
                "enthalpy_l": float(y[9] * scale),
                "Q_external": float(y[10] * scale),
            }
        )
    return {
        "volume": float(volume),
        "initial_energy": float(energy0),
        "samples": samples,
        "events": events,
        "response": None,
    }


def solve(payload):
    _validate(payload)
    if payload["operation"] == "equilibrium":
        return _static(payload)
    theta = float(payload.get("theta", 1.0))
    result = (
        _cycle(payload, theta, rtol=1e-9, max_step=2.0)
        if payload.get("response", False)
        else _cycle(payload, theta)
    )
    if payload.get("response", False):
        # A local differentiable neighbourhood need not contain an arbitrary
        # fixed relative perturbation.  Preserve the actual control branch,
        # allowing an interior one-sided stencil near a finite-time boundary.
        baseline = result
        signature = _event_signature(baseline)
        nominal_ki = float(payload["wall"]["Ki"])
        effective_ki = theta * nominal_ki
        cache = {effective_ki: baseline}

        def branch(value):
            if value not in cache and value <= 0:
                return None
            if value not in cache:
                # Sample physical conductance, then apply
                # d/dtheta = Ki_nominal*d/dKi_effective. A tiny positive theta
                # does not require subtracting unresolved theta perturbations.
                evaluated = dict(payload)
                evaluated["wall"] = dict(payload["wall"], Ki=value)
                try:
                    cache[value] = _cycle(evaluated, 1.0, rtol=1e-9, max_step=2.0)
                except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
                    cache[value] = None
            candidate = cache[value]
            return (
                candidate
                if candidate is not None and _event_signature(candidate) == signature
                else None
            )

        def estimate(step):
            lower, upper = branch(effective_ki - step), branch(effective_ki + step)
            if lower is not None and upper is not None:
                return (
                    nominal_ki
                    * (_response_values(upper) - _response_values(lower))
                    / (2 * step)
                )
            for direction, inside in ((1, upper), (-1, lower)):
                if inside is None:
                    continue
                farther = branch(effective_ki + direction * 2 * step)
                if farther is not None:
                    return (
                        nominal_ki
                        * direction
                        * (
                            -3 * _response_values(baseline)
                            + 4 * _response_values(inside)
                            - _response_values(farther)
                        )
                        / (2 * step)
                    )
            return None

        initial = payload["initial"]
        initial_phase = make_flasher().flash(
            T=initial["T"], P=initial["P"], zs=initial["z"]
        )
        fluid_capacity = initial["N"] * abs(float(initial_phase.Cv()))
        wall_capacity = payload["wall"]["C"]
        smaller = min(wall_capacity, fluid_capacity)
        larger = max(wall_capacity, fluid_capacity)
        interaction_capacity = smaller / (1 + smaller / larger)
        conductance_scale = interaction_capacity / payload["schedule"][-1]["until"]
        # Start above flash/integration noise, then halve until convergence.
        # A very small initial stencil can amplify phase-boundary roundoff;
        # repeatedly shrinking that stencil only worsens the derivative.
        step = 0.01 * max(effective_ki, conductance_scale)
        coarse = estimate(step)
        accepted = None
        for _ in range(16):
            step *= 0.5
            if (
                effective_ki + step == effective_ki
                or effective_ki - step == effective_ki
            ):
                break
            fine = estimate(step)
            if (
                fine is not None
                and coarse is not None
                and np.isfinite(fine).all()
                and np.isfinite(coarse).all()
            ):
                # Ten percent of the published acceptance bands leaves a
                # margin for independent thermodynamic/numerical approaches.
                tolerance = np.r_[
                    np.maximum(3e-5, 0.003 * np.abs(fine[:4])),
                    np.maximum(0.005, 0.003 * np.abs(fine[4:])),
                ]
                if np.all(np.abs(fine - coarse) <= tolerance):
                    accepted = fine
                    break
            coarse = fine
        if accepted is None:
            # Failure to resolve a derivative numerically is not evidence that
            # the caller's mathematical response is unsupported.
            raise RuntimeError("response resolution did not converge")
        result = baseline
        result["response"] = {
            "product_v": accepted[:2].tolist(),
            "product_l": accepted[2:4].tolist(),
            "event_times": accepted[4:].tolist(),
        }
    return result


def _event_signature(result):
    return [(e["phase"], e["kind"], e["cause"]) for e in result["events"]]


def _response_values(result):
    final = result["samples"][-1]
    return np.r_[
        final["product_v"],
        final["product_l"],
        [event["time"] for event in result["events"]],
    ]
