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
    
    # Dimension arrow for 'b' (vertical depth)
    ax.annotate('', xy=(a + 0.9, -b), xytext=(a + 0.9, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(a + 1.4, -b / 2, r'$b$', fontsize=11, ha='center', va='center')
    
    # Dimension arrow for '2a' (horizontal width)
    ax.annotate('', xy=(a, -b - 0.8), xytext=(-a, -b - 0.8),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(0, -b - 1.5, r'$2a$', fontsize=11, ha='center', va='center')

    ax.set_xlim(x_range)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Defect cross-section')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_nv_sensor(ax, x_pos, y_pos, theta, label_B_parallel=True):
    """
    Draw a schematic NV-center sensor at (x_pos, y_pos) with quantization
    axis rotated by angle theta from the surface normal (y-axis).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    x_pos, y_pos : float
        Sensor position coordinates (mm)
    theta : float
        Quantization axis rotation angle from vertical (radians)
    label_B_parallel : bool
        If True, label the B_‖ component at the sensor location
    """
    # Cross marker for sensor location
    cross_size = 0.2
    ax.plot([x_pos - cross_size, x_pos + cross_size], 
            [y_pos, y_pos], 'k-', lw=1.5, zorder=1)
    ax.plot([x_pos, x_pos], 
            [y_pos - cross_size, y_pos + cross_size], 'k-', lw=1.5, zorder=1)
    
    # Quantization axis: dotted line inclined by theta from vertical
    axis_len = 2.0
    axis_end_up_x = x_pos + axis_len * np.sin(theta)
    axis_end_up_y = y_pos + axis_len * np.cos(theta)
    axis_end_dn_x = x_pos - axis_len * np.sin(theta)
    axis_end_dn_y = y_pos - axis_len * np.cos(theta)
    ax.plot([axis_end_dn_x, axis_end_up_x], [axis_end_dn_y, axis_end_up_y],
            'k:', lw=1.5, zorder=2)
    
    # Arrow along quantization axis (inclined by theta) = B_parallel
    arrow_len = 1.5
    arrow_end_x = x_pos + arrow_len * np.sin(theta)
    arrow_end_y = y_pos + arrow_len * np.cos(theta)
    ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(x_pos, y_pos),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5, zorder=3))
    
    # Thin field-component arrows Hx (horizontal) and Hy (vertical),
    # both starting at the sensor cross; their tip-to-tail composition
    # forms B_parallel along the dotted axis. Hx extends to the B_parallel
    # x-coordinate and Hy's tip sits at the same y as the B_parallel arrow
    # tip.
    b_par_rel_x = 1.55 * np.sin(theta) + 0.1
    hy_len = arrow_len * np.cos(theta)
    ax.annotate('', xy=(x_pos + b_par_rel_x, y_pos), xytext=(x_pos, y_pos),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                zorder=3))
    ax.annotate('', xy=(x_pos, y_pos + hy_len), xytext=(x_pos, y_pos),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                zorder=3))
    ax.text(x_pos + b_par_rel_x / 2, y_pos - 0.15, r'$H_x$', fontsize=9,
            color='gray', ha='center', va='top', zorder=6)
    ax.text(x_pos - 0.15, y_pos + hy_len / 2, r'$H_y$',
            fontsize=9, color='gray', ha='right', va='center', zorder=6)
    
    # Angle annotation arc (theta from vertical)
    arc_radius = 0.6
    arc_angles = np.linspace(np.pi/2 - theta, np.pi/2, 50)
    arc_x = x_pos + arc_radius * np.cos(arc_angles)
    arc_y = y_pos + arc_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'b-', lw=1, alpha=0.8, zorder=4)
    
    # Theta label position
    theta_label_pos = (x_pos + 0.85 * np.sin(theta/2), 
                       y_pos + 0.95 * np.cos(theta/2))
    ax.text(theta_label_pos[0], theta_label_pos[1], r'$\theta$', 
            fontsize=11, color='blue', ha='center', va='center', zorder=5)
    
    # B_parallel label - upper part of the dotted axis
    if label_B_parallel:
        ax.text(x_pos + b_par_rel_x, y_pos + 1.55 * np.cos(theta) - 0.6, r'$B_{\parallel}$',
                fontsize=11, color='darkred', rotation=np.degrees(theta),
                ha='center', va='center', zorder=7)
