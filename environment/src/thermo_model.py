"""Frozen upstream PR model construction; no dynamic vessel implementation."""

import json
from pathlib import Path

from thermo import (
    CEOSGas,
    CEOSLiquid,
    PRMIX,
    ChemicalConstantsPackage,
    FlashVL,
    HeatCapacityGas,
    PropertyCorrelationsPackage,
)


def make_flasher():
    data = json.loads(Path(__file__).with_name("property_data.json").read_text())
    constants = ChemicalConstantsPackage(
        Tcs=data["Tcs"],
        Pcs=data["Pcs"],
        omegas=data["omegas"],
        MWs=data["MWs"],
        Tms=data["Tms"],
        Tbs=data["Tbs"],
        CASs=data["CASs"],
        names=data["species"],
    )
    cps = [
        HeatCapacityGas(poly_fit=tuple(bounds + [coeffs]))
        for bounds, coeffs in zip(
            data["cp_ranges"], data["cp_coefficients_descending"], strict=True
        )
    ]
    correlations = PropertyCorrelationsPackage(
        constants, HeatCapacityGases=cps, skip_missing=True
    )
    eos = {key: data[key] for key in ("Tcs", "Pcs", "omegas", "kijs")}
    gas = CEOSGas(PRMIX, eos, HeatCapacityGases=cps, T=300.0, P=1e6, zs=[0.5, 0.5])
    liquid = CEOSLiquid(
        PRMIX, eos, HeatCapacityGases=cps, T=300.0, P=1e6, zs=[0.5, 0.5]
    )
    return FlashVL(constants, correlations, liquid=liquid, gas=gas)
