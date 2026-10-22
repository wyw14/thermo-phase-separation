"""Finite-wall two-phase withdrawal cycle, a new extension of thermo 0.6.1.

Implement solve(payload) under the public task instruction.
The immutable PR parameters and a native FlashVL factory are available in
thermo_model.py.  thermo already supplies TP and VU flash; the missing work is
the dynamic process, inventory-control history, outputs and total response.
"""
from thermo_model import make_flasher


def solve(payload):
    raise NotImplementedError("Implement the public equilibrium/cycle interface")
