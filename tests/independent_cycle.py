"""Independent check: direct PR phase equations and RK45 integration.

This implementation never imports thermo or the reference solver.  It is used
on strictly two-phase trajectories by default. The optional vapor continuation
also covers the specified VL/V histories; it is not a global UV root enumerator.
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import expit, logit

sys.path.insert(0, "/tests")
from independent_pr import tp, phase


def cycle(p, theta=1.0, max_step=0.5, rtol=2e-9, allow_vapor=False):
    start = p["initial"]
    wall = p["wall"]
    init = tp(start["T"], start["P"], start["z"])
    volume = start["N"] * init["v"]
    energy = start["N"] * init["u"]
    n0 = np.array(start["z"]) * start["N"]
    y0 = np.r_[n0, energy / 10000, start["Tw"], np.zeros(7)]
    guess = np.array(
        [
            start["T"] / 300,
            np.log(start["P"]),
            logit(init["liquid"]["z"][0]),
            logit(init["gas"]["z"][0]),
            init["beta"],
        ]
    )
    last = {"state": None, "q": None}

    def equilibrium(y):
        nonlocal guess
        key = tuple(y[:3])
        if key == last["state"]:
            return last["q"]
        nt = sum(y[:2])
        zz = y[0] / nt
        vv = volume / nt
        uu = y[2] * 10000 / nt

        def equations(w, return_state=False):
            T = w[0] * 300
            P = np.exp(w[1])
            x0 = expit(w[2])
            y0 = expit(w[3])
            beta = w[4]
            x = np.array([x0, 1 - x0])
            yp = np.array([y0, 1 - y0])
            lp = phase(T, P, x, "l")
            vp = phase(T, P, yp, "v")
            if return_state:
                return T, P, beta, x, yp, lp, vp
            return np.r_[
                np.log(x) + lp["logphi"] - np.log(yp) - vp["logphi"],
                5 * ((1 - beta) * x0 + beta * y0 - zz),
                np.log(((1 - beta) * lp["v"] + beta * vp["v"]) / vv),
                ((1 - beta) * lp["u"] + beta * vp["u"] - uu) / 20000,
            ]

        fitted = None
        try:
            trial = root(equations, guess, tol=1e-10)
            if np.linalg.norm(equations(trial.x), ord=np.inf) <= 2e-8:
                fitted = trial
        except (ArithmeticError, ValueError, TypeError, IndexError):
            pass
        q = equations(fitted.x, True) if fitted is not None else None
        if q is not None and 0 <= q[2] <= 1 and q[4][0] - q[3][0] >= 0.04:
            guess = fitted.x
        elif allow_vapor:
            composition = np.array([zz, 1-zz])

            def gas_residual(w):
                T, P = w[0]*300, np.exp(w[1])
                props = phase(T, P, composition, "v")
                return [np.log(props["v"]/vv), (props["u"]-uu)/20000]

            gas_fit = root(gas_residual, guess[:2], tol=1e-10)
            if np.linalg.norm(gas_residual(gas_fit.x), ord=np.inf) > 2e-8:
                raise RuntimeError("independent vapor UV residual")
            T, P = gas_fit.x[0]*300, np.exp(gas_fit.x[1])
            stable_tp = tp(T, P, composition)
            if stable_tp["gas"] is not None and stable_tp["liquid"] is not None:
                raise RuntimeError("independent vapor fails TP stability")
            vp = phase(T, P, composition, "v")
            q = (T, P, 1.0, None, composition, None, vp)
            guess[:2] = gas_fit.x
        else:
            raise RuntimeError("independent checker limited to VL interior")
        last["state"], last["q"] = key, q
        return q

    enabled_state = [False, False]
    events = []
    pieces = []
    time = 0.0
    y = y0
    thresholds = [p["controllers"][s] for s in ("v", "l")]

    def update(t, y, commands, cause, triggered=None):
        q = equilibrium(y)
        nv = sum(y[:2]) * q[2]
        amount = [nv, sum(y[:2]) - nv]
        old = list(enabled_state)
        triggered = {} if triggered is None else triggered
        for j in range(2):
            if not commands[j]:
                enabled_state[j] = False
            elif j in triggered:
                enabled_state[j] = triggered[j]
            elif old[j] and amount[j] <= thresholds[j]["off"]:
                enabled_state[j] = False
            elif not old[j] and amount[j] >= thresholds[j]["on"]:
                enabled_state[j] = True
        changed = []
        for j in range(2):
            if old[j] != enabled_state[j]:
                changed.append(
                    {
                        "time": float(t),
                        "phase": "v" if j == 0 else "l",
                        "kind": "open" if enabled_state[j] else "close",
                        "before": old[j],
                        "after": enabled_state[j],
                        "cause": cause,
                    }
                )
        events.extend(sorted(changed, key=lambda e: (e["after"], e["phase"] != "v")))

    at_boundary = {}
    for seg in p["schedule"]:
        update(time, y, seg["enabled"], "command", at_boundary)
        at_boundary = {}
        while time < seg["until"]:
            mode = tuple(enabled_state)

            def derivative(t, y):
                T, P, beta, x, yp, lp, vp = equilibrium(y)
                flows = [seg["flow"][j] if mode[j] else 0.0 for j in range(2)]
                pv = flows[0] * yp if vp is not None else np.zeros(2)
                pl = flows[1] * x if lp is not None else np.zeros(2)
                hv = flows[0] * vp["h"] if vp is not None else 0.0
                hl = flows[1] * lp["h"] if lp is not None else 0.0
                qi = theta * wall["Ki"] * (y[3] - T)
                qo = wall["Ko"] * (seg["ambient"] - y[3])
                return np.r_[
                    -pv - pl,
                    (qi - hv - hl) / 10000,
                    (qo - qi) / wall["C"],
                    pv,
                    pl,
                    hv / 10000,
                    hl / 10000,
                    qo / 10000,
                ]

            watches, channels = [], []
            for j in range(2):
                if not seg["enabled"][j]:
                    continue
                threshold = thresholds[j]["off" if mode[j] else "on"]

                def watch(t, y, j=j, threshold=threshold):
                    beta = equilibrium(y)[2]
                    nt = sum(y[:2])
                    return nt * (beta if j == 0 else 1 - beta) - threshold

                watch.terminal = True
                watch.direction = -1 if mode[j] else 1
                watches.append(watch)
                channels.append(j)
            out = solve_ivp(
                derivative,
                (time, seg["until"]),
                y,
                method="RK45",
                rtol=rtol,
                atol=1e-10,
                max_step=max_step,
                events=watches,
                dense_output=True,
            )
            if not out.success:
                raise RuntimeError(out.message)
            new = float(out.t[-1])
            pieces.append((time, new, out.sol, mode))
            at_boundary = {
                channels[i]: not mode[channels[i]]
                for i, located in enumerate(out.t_events)
                if len(located)
            }
            if new < time or (new == time and not at_boundary):
                raise RuntimeError("independent event integration did not advance")
            time = new
            y = out.y[:, -1]
            if time < seg["until"]:
                previous = tuple(enabled_state)
                update(time, y, seg["enabled"], "inventory", at_boundary)
                at_boundary = {}
                if tuple(enabled_state) == previous:
                    raise RuntimeError("independent event did not change control")
    update(time, y, p["schedule"][-1]["enabled"], "inventory", at_boundary)
    samples = []
    for t in p["times"]:
        a, b, interp, mode = [s for s in pieces if s[0] <= t <= s[1]][-1]
        yy = interp(t)
        mode = [False, False]
        for event in events:
            if event["time"] > t:
                break
            mode[0 if event["phase"] == "v" else 1] = event["after"]
        T, P, beta, x, yp, lp, vp = equilibrium(yy)
        nt = sum(yy[:2])
        samples.append(
            {
                "time": t,
                "T": T,
                "P": P,
                "Tw": float(yy[3]),
                "n": yy[:2].tolist(),
                "U": float(yy[2] * 10000),
                "Nv": nt * beta,
                "Nl": nt * (1 - beta),
                "y": yp.tolist() if yp is not None else None,
                "x": x.tolist() if x is not None else None,
                "on": list(mode),
                "product_v": yy[4:6].tolist(),
                "product_l": yy[6:8].tolist(),
                "enthalpy_v": float(yy[8] * 10000),
                "enthalpy_l": float(yy[9] * 10000),
                "Q_external": float(yy[10] * 10000),
            }
        )
    return {
        "volume": float(volume),
        "initial_energy": float(energy),
        "samples": samples,
        "events": events,
        "response": None,
    }
