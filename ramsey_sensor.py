"""
Two-level quantum sensor: NV(-) ms=0 <-> ms=-1 Ramsey sequence.

Two equivalent ways to evaluate the Ramsey transition probability are
provided:

* `ramsey_p1_qutip` builds the pi/2 - free evolution - pi/2 sequence
  explicitly as 2x2 unitaries with QuTiP and applies them to the
  initial state, exactly mirroring the physical pulse sequence.
* `ramsey_p1` is the closed form obtained by working that same
  circuit out analytically. It is used for all the parameter sweeps
  in main_simulation.py because it is orders of magnitude faster than
  building a QuTiP circuit at every point; `main_simulation.py`
  cross-checks the two against each other to confirm they agree.

Working the circuit out by hand: starting from |psi0> = |0>,

    |psi0> --Rx(pi/2)--> (|0> - i|1>)/sqrt(2)
           --Rz(phi)-->  (e^{-i phi/2}|0> - i e^{i phi/2}|1>)/sqrt(2)
           --Rx(pi/2)--> ... ,   phi = gamma * B_par * tau,

gives a final amplitude on |1> of -i*cos(phi/2), so

    P1(tau) = |<1|psi_f>|^2 = cos^2(phi/2) = 0.5 * (1 + cos(phi)).

Field: microtesla (uT). Time: microseconds (us).
"""

import numpy as np
from qutip import basis, sigmax, sigmaz, expect

# Electron-spin gyromagnetic ratio of the NV ground state
# (gamma_e/2pi = 28 GHz/T = 28 kHz/uT), in units of rad / (us . uT).
GAMMA_NV = 2 * np.pi * 28e-3

_PSI0 = basis(2, 0)
_U_PI2 = (-1j * (np.pi / 4) * sigmax()).expm()  # pi/2 pulse about x


def project_field(Hx, Hy, theta):
    """
    Component of the leakage field along a sensor axis tilted by
    `theta` from the surface normal.

    theta = 0      -> axis along y (surface normal): senses Hy only
    theta = pi / 2 -> axis along x (tangential): senses Hx only
    """
    return Hx * np.sin(theta) + Hy * np.cos(theta)


def ramsey_phase(B_par, tau, gamma=GAMMA_NV):
    """Accumulated Ramsey phase, phi = gamma * B_par * tau."""
    return gamma * B_par * tau


def ramsey_p1(B_par, tau, gamma=GAMMA_NV):
    """
    Closed-form Ramsey transition probability
        P1(tau) = 0.5 * (1 + cos(phi)),   phi = gamma * B_par * tau.
    Vectorized: B_par and/or tau may be arrays.
    """
    phi = ramsey_phase(B_par, tau, gamma)
    return 0.5 * (1.0 + np.cos(phi))


def ramsey_p1_qutip(B_par, tau, gamma=GAMMA_NV):
    """
    Ramsey transition probability, computed explicitly with QuTiP:
    pi/2 pulse - free precession under H = (gamma/2) B_par sigma_z for
    time tau - pi/2 pulse - measure P(ms=-1). Scalar B_par, tau only;
    used solely as a cross-check against `ramsey_p1` (see
    main_simulation.py), not for the parameter sweeps themselves.
    """
    H_free = 0.5 * gamma * B_par * sigmaz()
    U_free = (-1j * H_free * tau).expm()
    psi_f = _U_PI2 * U_free * _U_PI2 * _PSI0
    return expect(0.5 * (1 - sigmaz()), psi_f)
