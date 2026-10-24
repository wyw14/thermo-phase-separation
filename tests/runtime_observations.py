"""Trusted transport failures observed outside the candidate process."""

_FAULTS = []


def record_fault(kind, operation):
    """Record only an actual timeout, crash, bootstrap or protocol failure."""
    if kind not in {"timeout", "crash", "protocol", "bootstrap"}:
        raise ValueError("Unknown runtime fault kind")
    _FAULTS.append({"kind": kind, "operation": operation})


def snapshot():
    """Supply structured faults to the trusted completion ledger."""
    return {"runtime_faults": list(_FAULTS)}
