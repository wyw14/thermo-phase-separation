"""Independent PR cubic/fugacity/caloric equations, without importing thermo.

Constants and heat-capacity polynomials are the public upstream fixture.  This
calculator deliberately does not implement the candidate's vessel integrator.
"""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

R = 8.31446261815324
SQRT2 = np.sqrt(2.0)
DATA = json.loads((Path(__file__).parent / "data" / "property_data.json").read_text())


def phase(T, P, z, kind):
    """Compute one PR phase from independently evaluated cubic roots."""
    z = np.asarray(z, dtype=float)
    tc, pc, omega = [np.asarray(DATA[x]) for x in ("Tcs", "Pcs", "omegas")]
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    ac = 0.4572355289213822 * R**2 * tc**2 / pc
    bvec = 0.07779607390388846 * R * tc / pc
    base = 1 + kappa * (1 - np.sqrt(T / tc))
    avec = ac * base**2
    davec = -ac * base * kappa / np.sqrt(T * tc)
    aij = np.sqrt(np.outer(avec, avec))
    daij = 0.5 * aij * (davec[:, None] / avec[:, None] + davec[None, :] / avec[None, :])
    a, da, b = float(z @ aij @ z), float(z @ daij @ z), float(z @ bvec)
    A, B = a * P / (R * T) ** 2, b * P / (R * T)
    roots = np.roots([1, B - 1, A - 3 * B * B - 2 * B, -A * B + B * B + B**3])
    real = sorted(float(q.real) for q in roots if abs(q.imag) < 1e-7 and q.real > B)
    Z = real[-1] if kind == "v" else real[0]
    logterm = np.log((Z + (1 + SQRT2) * B) / (Z + (1 - SQRT2) * B))
    logphi = (
        bvec / b * (Z - 1)
        - np.log(Z - B)
        - A / (2 * SQRT2 * B) * (2 * (aij @ z) / a - bvec / b) * logterm
    )
    h_ideal = 0.0
    for zi, coeff in zip(z, DATA["cp_coefficients_descending"], strict=True):
        primitive = np.polyint(coeff)
        h_ideal += zi * (np.polyval(primitive, T) - np.polyval(primitive, 298.15))
    h = h_ideal + R * T * (Z - 1) + (T * da - a) / (2 * SQRT2 * b) * logterm
    v = Z * R * T / P
    physical_kind = kind
    if len(real) == 1:
        # The gas/liquid constructors share this root. Choose its physical
        # label from the PR pressure derivatives, rather than list/tie order.
        denominator = v*v + 2*b*v - b*b
        denominator_prime = 2*(v+b)
        dp_dt = R/(v-b) - da/denominator
        dp_dv = -R*T/(v-b)**2 + a*denominator_prime/denominator**2
        d2p_dtdv = -R/(v-b)**2 + da*denominator_prime/denominator**2
        d2p_dv2 = 2*R*T/(v-b)**3 + a*(2/denominator**2-2*denominator_prime**2/denominator**3)
        identification = v*(d2p_dtdv/dp_dt-d2p_dv2/dp_dv)
        physical_kind = "l" if identification > 1.00000000000001 else "v"
    return {"v": v, "h": h, "u": h - P * v, "logphi": logphi, "z": z, "kind": physical_kind}


def tp(T, P, z):
    """Choose stable one/two-phase TP state by common fugacity and Gibbs energy."""
    z = np.asarray(z, dtype=float)
    candidates = []
    for kind in ("v", "l"):
        q = phase(T, P, z, kind)
        kind = q["kind"]
        gibbs = float(z @ (np.log(z) + q["logphi"]))
        candidates.append(
            (
                gibbs,
                1.0 if kind == "v" else 0.0,
                q if kind == "v" else None,
                q if kind == "l" else None,
            )
        )
    eps = 1e-8

    def residual(values):
        x = np.array([values[0], 1 - values[0]])
        y = np.array([values[1], 1 - values[1]])
        lp = phase(T, P, x, "l")
        vp = phase(T, P, y, "v")
        return np.log(x) + lp["logphi"] - np.log(y) - vp["logphi"]

    for xl, yv in ((0.1, 0.9), (0.3, 0.85), (0.45, 0.99)):
        guess = [
            min(z[0] * 0.9, max(eps * 2, xl)),
            max(z[0] + 0.01, min(1 - eps * 2, yv)),
        ]
        fit = least_squares(
            residual,
            guess,
            bounds=([eps, z[0] + eps], [z[0] - eps, 1 - eps]),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=150,
        )
        x0, y0 = fit.x
        if np.linalg.norm(residual(fit.x), ord=np.inf) > 1e-7 or y0 - x0 < 0.04:
            continue
        beta = (z[0] - x0) / (y0 - x0)
        x = np.array([x0, 1 - x0])
        y = np.array([y0, 1 - y0])
        lp, vp = phase(T, P, x, "l"), phase(T, P, y, "v")
        gibbs = float(
            (1 - beta) * (x @ (np.log(x) + lp["logphi"]))
            + beta * (y @ (np.log(y) + vp["logphi"]))
        )
        candidates.append((gibbs, beta, vp, lp))
    _, beta, vp, lp = min(candidates, key=lambda row: row[0])
    return {
        "beta": beta,
        "gas": vp,
        "liquid": lp,
        "v": beta * (vp["v"] if vp else 0) + (1 - beta) * (lp["v"] if lp else 0),
        "u": beta * (vp["u"] if vp else 0) + (1 - beta) * (lp["u"] if lp else 0),
    }
