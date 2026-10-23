# Finite-wall phase separator

This application extension uses [thermo 0.6.1](https://github.com/CalebBell/thermo/tree/4f51c4343773c8e50480b32eb81b014ae3945ebb), distributed under the MIT license. The fixed ethane/n-pentane PR parameters and heat-capacity polynomials come from its `tests/test_flash_vl.py:test_C2_C5_PR` fixture. The upstream license is included in `environment/src/LICENSE-thermo.txt`. Process inputs are synthetic.

Implement `/app/src/candidate.py:solve`. The physical model and interface are defined in `instruction.md`; `thermo_model.make_flasher()` constructs the fixed native model. Runtime dependencies are installed from the bundled, hash-checked wheels.

The separate verifier requires Linux/amd64 and root bootstrap. Candidate calculations use a nonroot identity, private temporary storage and no network. Filesystem isolation uses Landlock when available or a dependency-only chroot with standard Docker capabilities (`SYS_CHROOT`, `MKNOD`, `SETUID`, `SETGID`). Tests and reference code remain inaccessible to candidate calculations. A privileged container is unnecessary.

The task uses standard Harbor entry points:

```sh
harbor run -p /path/to/task -a oracle
harbor run -p /path/to/task -a nop
```
