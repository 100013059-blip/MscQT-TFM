"""
Demonstration: generates Figures 1-4 of Chapter 5.

This script adds the bare minimum needed to get a Ramsey MFL
measurement that actually reproduces the theoretical field shapes
from Section "Expected results", instead of the distorted, folded-
over signal a poorly chosen free-evolution time produces:

1. Operating-point rule (tau_scan, B_bias)
   ----------------------------------------
   The fringe P1(phi) = 0.5*(1+cos(phi)) is single-valued (monotonic)
   in the field only while phi stays within one branch of the cosine.
   If tau is too large, the phase gamma*B(x)*tau folds back over a
   maximum/minimum of the fringe as the sensor is scanned across the
   defect, and the readout stops resembling the field shape (e.g. a
   single peak in Hx can turn into two separate bumps in P1 -- exactly
   the artifact obtained with an untuned tau). To avoid this, tau is
   chosen so the *peak* defect field only accumulates a modest phase
   swing, phi_target = pi/4, around a bias point sitting at quadrature
   (phi_bias = pi/2, the steepest and most sensitive part of the
   fringe):

       tau_scan = phi_target / (gamma * B_max)
       B_bias   = (pi/2) / (gamma * tau_scan)

   This keeps the total phase within [pi/4, 3pi/4] -- a monotonic,
   reasonably linear stretch of the fringe -- across the whole scan,
   for any orientation and any field magnitude up to B_max.

2. Correctness check
   ------------------
   `ramsey_p1` (closed form) and `ramsey_p1_qutip` (explicit QuTiP
   circuit) are cross-checked below; they must agree to numerical
   precision, confirming the fringe used for the fast sweeps is the
   same one the underlying pulse sequence produces.
"""

import numpy as np
import matplotlib.pyplot as plt

from defect_field import notch_leakage_field, plot_notch_geometry, plot_nv_sensor
from ramsey_sensor import (
    GAMMA_NV, project_field,
    ramsey_p1, ramsey_p1_qutip,
)


def main():
    # --- defect + scan geometry ---
    a, b, liftoff = 3.0, 2.0, 0.3          # mm
    rho_m = 6.0                             # sets field scale (a few uT peak)
    x = np.linspace(-15, 15, 400)
    Hx, Hy = notch_leakage_field(x, a, b, liftoff, rho_m)
    B_max = max(np.max(np.abs(Hx)), np.max(np.abs(Hy)))
    print(f"Peak |Hx| = {np.max(np.abs(Hx)):.2f} uT, "
          f"peak |Hy| = {np.max(np.abs(Hy)):.2f} uT")

    # --- Figure 1: geometry + field profile + liftoff comparison ---
    fig1, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    
    # Panel 1: Defect geometry
    plot_notch_geometry(ax[0], a=a, b=b)
    
    # Panel 2: Field profile at baseline liftoff
    ax[1].plot(x, Hx, 'b-', lw=2, label=r'Tangential $H_x$ (symmetric)')
    ax[1].plot(x, Hy, 'r--', lw=2, label=r'Normal $H_y$ (antisymmetric)')
    ax[1].axhline(0, color='k', lw=0.7)
    ax[1].set_xlabel('Scan position x (mm)')
    ax[1].set_ylabel(r'Leakage field ($\mu$T)')
    ax[1].set_title('MFL field above the notch\n(Zatsepin-Shcherbinin)')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Liftoff comparison (Hx solid, Hy dashed)
    liftoffs = [0.3, 1.0, 2.0, 4.0]  # mm
    colors = ['darkorange', 'seagreen', 'steelblue', 'indigo']
    for lo, c in zip(liftoffs, colors):
        Hx_lo, Hy_lo = notch_leakage_field(x, a, b, lo, rho_m)
        ax[2].plot(x, Hx_lo, color=c, lw=1.6, ls='-')
        ax[2].plot(x, Hy_lo, color=c, lw=1.6, ls='--')
    ax[2].axhline(0, color='k', lw=0.7)
    ax[2].set_xlabel('Scan position x (mm)')
    ax[2].set_ylabel(r'Leakage field ($\mu$T)')
    ax[2].set_title('Liftoff dependence\n(solid=$H_x$, dashed=$H_y$)')
    
    # Legend: show liftoffs as color entries
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax[2].legend(custom_lines, [f'{lo:g} mm' for lo in liftoffs], 
                 title='Liftoff', title_fontsize=9, fontsize=8.5)
    ax[2].grid(True, linestyle='--', alpha=0.5)
    
    fig1.tight_layout()
    fig1.savefig('fig1_geometry_field.png', dpi=140)

    # --- Figure 1b: geometry with NV sensor schematic ---
    # Shows the defect cross-section with an NV-center quantum sensor
    # positioned at (x=0, y=liftoff) with its quantization axis rotated
    # by angle theta from the vertical. Labels B_‖ at the sensor.
    fig1b, ax1b = plt.subplots(figsize=(8, 7))
    
    # Draw the defect geometry
    plot_notch_geometry(ax1b, x_range=(-15, 15), a=a, b=b)
    
    # NV sensor parameters
    theta_sensor = np.pi / 4  # 45-degree rotation from vertical
    sensor_x = 0.0
    sensor_y = liftoff
    
    # Draw the NV sensor schematic
    plot_nv_sensor(ax1b, sensor_x, sensor_y, theta_sensor,
                   label_B_parallel=True)
    
    ax1b.set_title('MFL Defect Geometry with NV Quantum Sensor', fontsize=11)
    fig1b.tight_layout()
    fig1b.savefig('fig1_geometry_field_bis.png', dpi=140)

    # --- Operating point: pick tau_scan and B_bias from the peak field ---
    phi_target = np.pi / 4  # keeps the peak field within a monotonic,
                             # near-linear stretch of the fringe
    tau_scan = phi_target / (GAMMA_NV * B_max)
    B_bias = (np.pi / 2) / (GAMMA_NV * tau_scan)
    print(f"tau_scan = {tau_scan:.2f} us, B_bias = {B_bias:.2f} uT")

    # --- Correctness check: closed form vs explicit QuTiP circuit ---
    for Bv in (0.0, 0.5 * B_max, B_max, -B_max, B_bias):
        p_analytic = ramsey_p1(Bv, tau_scan)
        p_qutip = ramsey_p1_qutip(Bv, tau_scan)
        assert np.isclose(p_analytic, p_qutip, atol=1e-9), (
            f"Closed-form and QuTiP Ramsey formulas disagree at B={Bv} uT: "
            f"{p_analytic} vs {p_qutip}"
        )
    print("Ramsey fringe check passed: closed form matches the explicit "
          "QuTiP circuit to machine precision.")

    # --- Figure 2: Ramsey fringes vs free-evolution time tau ---
    # A fixed test field (the peak defect field) is used just to show
    # how the Ramsey signal oscillates as tau is increased. The fringe
    # period is T = 2*pi / (gamma * B_test); the sweep is extended to
    # cover several periods so the plot actually looks like a fringe
    # pattern rather than a single half-oscillation. The closed form
    # (curve) is overlaid with sparse explicit-QuTiP-circuit points
    # (markers) as a visual confirmation that the two agree.
    B_test = B_max
    n_periods = 4
    T_fringe = 2 * np.pi / (GAMMA_NV * B_test)
    tau_max = n_periods * T_fringe
    tau_grid = np.linspace(0.0, tau_max, 600)
    p1_curve = ramsey_p1(B_test, tau_grid)
    tau_markers = np.linspace(0.0, tau_max, 41)
    p1_markers = np.array([ramsey_p1_qutip(B_test, t) for t in tau_markers])

    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    ax2.plot(tau_grid, p1_curve, 'k-', lw=1.6, label='closed form')
    ax2.plot(tau_markers, p1_markers, 'o', color='crimson', ms=4,
              label='explicit QuTiP circuit')
    ax2.axvline(tau_scan, color='gray', ls=':', lw=1.2,
                label=rf'$\tau_\mathrm{{scan}}$={tau_scan:.2f} $\mu$s')
    ax2.set_xlabel(r'Free evolution time $\tau$ ($\mu$s)')
    ax2.set_ylabel(r'$P_{|1\rangle}$')
    ax2.set_title(rf'Ramsey fringes for a fixed field $B$={B_test:.2f} $\mu$T '
                    rf'(period $T=2\pi/\gamma B$={T_fringe:.1f} $\mu$s)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()
    fig2.savefig('fig2_ramsey_fringes.png', dpi=140)

    # --- Figure 3: spatial defect scan with two sensor orientations ---
    # tau_scan and B_bias come from the operating-point rule above, so
    # the readout stays in a monotonic, near-linear part of the fringe
    # across the whole scan.
    B_normal = B_bias + project_field(Hx, Hy, theta=0.0)        # senses Hy
    B_tangent = B_bias + project_field(Hx, Hy, theta=np.pi / 2)  # senses Hx

    P1_normal_true = ramsey_p1(B_normal, tau_scan)
    P1_tangent_true = ramsey_p1(B_tangent, tau_scan)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax3a.plot(x, Hx, 'b-', lw=1.6, label=r'$H_x$ (tangential)')
    ax3a.plot(x, Hy, 'r--', lw=1.6, label=r'$H_y$ (normal)')
    ax3a.axhline(0, color='k', lw=0.6)
    ax3a.set_ylabel(r'Field ($\mu$T)')
    ax3a.set_title('True MFL field components')
    ax3a.legend()
    ax3a.grid(True, linestyle='--', alpha=0.5)

    ax3b.plot(x, P1_tangent_true, color='navy', lw=2,
              label=r'Sensor axis $\parallel$ surface (senses $H_x$)')
    ax3b.plot(x, P1_normal_true, color='darkorange', lw=2,
              label=r'Sensor axis $\perp$ surface (senses $H_y$)')
    ax3b.set_xlabel('Scan position x (mm)')
    ax3b.set_ylabel(r'$P_{|1\rangle}(\tau)$')
    ax3b.set_title(rf'Ramsey defect-detection signal at $\tau$={tau_scan:.2f} $\mu$s, '
                    rf'bias $B_{{bias}}$={B_bias:.2f} $\mu$T')
    ax3b.legend(fontsize=8.5)
    ax3b.grid(True, linestyle='--', alpha=0.5)
    fig3.tight_layout()
    fig3.savefig('fig3_spatial_scan.png', dpi=140)

    # --- Figure 4: effect of increasing the sensor liftoff ---
    # Everything else (defect geometry, tau_scan, B_bias) stays fixed;
    # only the sensor's height above the surface is varied. Raising the
    # sensor moves it further from the magnetic "charge sheets" at the
    # notch walls, so the field it samples gets weaker and more spread
    # out in x -- the classic MFL lift-off effect.
    liftoffs = [0.3, 1.0, 2.0, 4.0]  # mm
    colors = ['darkorange', 'seagreen', 'steelblue', 'indigo']

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for lo, c in zip(liftoffs, colors):
        Hx_lo, Hy_lo = notch_leakage_field(x, a, b, lo, rho_m)
        B_normal_lo = B_bias + project_field(Hx_lo, Hy_lo, theta=0.0)
        B_tangent_lo = B_bias + project_field(Hx_lo, Hy_lo, theta=np.pi / 2)
        P1_normal_lo = ramsey_p1(B_normal_lo, tau_scan)
        P1_tangent_lo = ramsey_p1(B_tangent_lo, tau_scan)

        ax4a.plot(x, Hx_lo, color=c, lw=1.8, ls='-', 
                  label=rf'{lo:g} mm ' if lo == liftoffs[0] else None)
        ax4a.plot(x, Hy_lo, color=c, lw=1.8, ls='--', 
                  label=rf'{lo:g} mm' if lo == liftoffs[0] else None)
        ax4b.plot(x, P1_tangent_lo, color=c, lw=2, ls='-',
                  label=rf'tangential ($H_x$), liftoff={lo:g} mm' if lo == liftoffs[0] else None)
        ax4b.plot(x, P1_normal_lo, color=c, lw=2, ls='--',
                  label=rf'normal ($H_y$), liftoff={lo:g} mm' if lo == liftoffs[0] else None)

    ax4a.axhline(0, color='k', lw=0.6)
    ax4a.set_ylabel(r'Leakage field ($\mu$T)')
    ax4a.set_title('Effect of sensor liftoff on the true defect field')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax4a.legend(custom_lines, [f'{lo:g} mm' for lo in liftoffs], 
                title='Liftoff', title_fontsize=9, fontsize=8.5)
    ax4a.text(-14, 2.2, 'solid = H_x (tangential)', fontsize=8.5, 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4a.text(-14, 1.9, 'dashed = H_y (normal)', fontsize=8.5, 
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4a.grid(True, linestyle='--', alpha=0.5)

    ax4b.set_xlabel('Scan position x (mm)')
    ax4b.set_ylabel(r'$P_{|1\rangle}(\tau)$')
    ax4b.set_title(r'Ramsey signal for both sensor orientations at $\tau$={:.2f} $\mu$s '
                   'for increasing liftoff'.format(tau_scan))
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax4b.legend(custom_lines, [f'liftoff={lo:g} mm' for lo in liftoffs], 
                title='Orientation: solid=tangential (H_x), dashed=normal (H_y)',
                title_fontsize=8, fontsize=8.5)
    ax4b.grid(True, linestyle='--', alpha=0.5)
    fig4.tight_layout()
    fig4.savefig('fig4_liftoff_effect.png', dpi=140)

    plt.show()


if __name__ == "__main__":
    main()
