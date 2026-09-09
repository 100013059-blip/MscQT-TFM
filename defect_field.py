"""
Magnetostatic model of the defect (Zatsepin & Shcherbinin 1966).

Computes the tangential (Hx) and normal (Hy) magnetic-flux-leakage
(MFL) field components above a rectangular surface-breaking notch,
and draws the notch cross-section for the simulation figures.

`rho_m` is an arbitrary-units surface magnetic charge density (see
Eqs. (1)-(2) of Chapter 5); it simply sets the overall field scale.
It is chosen in main_simulation.py to give peak fields of order a
few microtesla, comparable to the near-surface fields reported for
small laboratory-scale defects in the MFL literature (e.g. Dutta,
Ghorbel & Stanley 2009), which is the regime an NV-diamond
magnetometer is designed to resolve.
"""

import numpy as np


def notch_leakage_field(x, a, b, liftoff, rho_m=1.0):
    """
    Tangential (Hx) and normal (Hy) MFL field components above a
    rectangular surface-breaking notch of half-width a and depth b,
    measured at sensor height `liftoff` above the surface.
    """
    x = np.asarray(x, dtype=float)
    dx_p = x + a
    dx_m = x - a
    y = liftoff

    Hx = (rho_m / (2 * np.pi)) * (
        np.arctan2(y + b, dx_p) - np.arctan2(y, dx_p)
        - np.arctan2(y + b, dx_m) + np.arctan2(y, dx_m)
    )

    num = (dx_p**2 + (y + b)**2) * (dx_m**2 + y**2)
    den = (dx_m**2 + (y + b)**2) * (dx_p**2 + y**2)
    Hy = (rho_m / (4 * np.pi)) * np.log(num / den)

    return Hx, Hy


def plot_notch_geometry(ax, x_range=(-10, 10), a=3.0, b=2.0):
    """Draw the 2D surface-breaking notch cross-section on `ax`."""
    y_bottom = -b - 2
    y_top = 2

    mat_x = np.array([x_range[0], x_range[1], x_range[1], a, a, -a, -a, x_range[0]])
    mat_y = np.array([y_bottom, y_bottom, 0, 0, -b, -b, 0, 0])
    ax.fill(mat_x, mat_y, color='0.75', label='Ferromagnetic material')

    ax.plot([-a, -a, a, a], [0, -b, -b, 0], 'k-', linewidth=2)
    ax.plot([x_range[0], -a], [0, 0], 'k-', linewidth=2.5)
    ax.plot([a, x_range[1]], [0, 0], 'k-', linewidth=2.5)
    ax.text(a + 0.5, -b / 2, f'b={b} mm', fontsize=9, va='center')
    ax.text(0, -b - 0.8, f'a={a} mm (half-width)', fontsize=9, ha='center')

    ax.set_xlim(x_range)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Defect cross-section')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
