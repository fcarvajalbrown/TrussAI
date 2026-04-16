from strands import tool
import numpy as np


@tool
def euler_beam(
    length: float,
    E: float,
    I: float,
    load_type: str,
    load_value: float,
    support_type: str,
) -> dict:
    """
    Analyze a beam using Euler-Bernoulli theory (ignores shear deformation).
    Valid for slender beams where L/d > 20.
    length: beam length in meters
    E: Young's modulus in Pa
    I: second moment of area in m^4
    load_type: 'point_center', 'point_end', 'uniform'
    load_value: load in N or N/m for uniform
    support_type: 'simply_supported', 'cantilever', 'fixed_fixed'
    """
    L = length
    EI = E * I
    delta_max = M_max = V_max = 0.0

    if support_type == "simply_supported":
        if load_type == "point_center":
            delta_max = load_value * L**3 / (48 * EI)
            M_max = load_value * L / 4
            V_max = load_value / 2
        elif load_type == "uniform":
            delta_max = 5 * load_value * L**4 / (384 * EI)
            M_max = load_value * L**2 / 8
            V_max = load_value * L / 2
        else:
            return {"status": "error", "message": f"Unsupported load_type '{load_type}' for simply_supported."}

    elif support_type == "cantilever":
        if load_type == "point_end":
            delta_max = load_value * L**3 / (3 * EI)
            M_max = load_value * L
            V_max = load_value
        elif load_type == "uniform":
            delta_max = load_value * L**4 / (8 * EI)
            M_max = load_value * L**2 / 2
            V_max = load_value * L
        else:
            return {"status": "error", "message": f"Unsupported load_type '{load_type}' for cantilever."}

    elif support_type == "fixed_fixed":
        if load_type == "point_center":
            delta_max = load_value * L**3 / (192 * EI)
            M_max = load_value * L / 8
            V_max = load_value / 2
        elif load_type == "uniform":
            delta_max = load_value * L**4 / (384 * EI)
            M_max = load_value * L**2 / 12
            V_max = load_value * L / 2
        else:
            return {"status": "error", "message": f"Unsupported load_type '{load_type}' for fixed_fixed."}

    else:
        return {"status": "error", "message": f"Unknown support_type '{support_type}'."}

    d_approx = 2 * np.sqrt(I)
    slenderness = L / d_approx if d_approx > 0 else float("inf")
    slenderness_note = (
        "Euler-Bernoulli valid (L/d > 20)"
        if slenderness > 20
        else f"WARNING: L/d = {slenderness:.1f} — consider Timoshenko for accuracy"
    )

    return {
        "status": "ok",
        "theory": "Euler-Bernoulli",
        "support_type": support_type,
        "load_type": load_type,
        "EI_Nm2": round(EI, 4),
        "delta_max_m": round(delta_max, 8),
        "M_max_Nm": round(M_max, 4),
        "V_max_N": round(V_max, 4),
        "slenderness_L_d": round(slenderness, 2),
        "slenderness_note": slenderness_note,
    }


@tool
def timoshenko_beam(
    length: float,
    E: float,
    I: float,
    G: float,
    A: float,
    kappa: float,
    load_type: str,
    load_value: float,
    support_type: str,
) -> dict:
    """
    Analyze a beam using Timoshenko theory (includes shear deformation).
    Best for short/thick beams where L/d <= 20.
    length: beam length in meters
    E: Young's modulus in Pa
    I: second moment of area in m^4
    G: shear modulus in Pa
    A: cross-section area in m^2
    kappa: shear correction factor (0.833 rectangular, 0.9 circular)
    load_type: 'point_center', 'point_end', 'uniform'
    load_value: load in N or N/m for uniform
    support_type: 'simply_supported', 'cantilever'
    """
    L = length
    EI = E * I
    kGA = kappa * G * A
    phi = 12 * EI / (kGA * L**2)

    delta_bending = delta_shear = M_max = V_max = 0.0

    if support_type == "simply_supported":
        if load_type == "point_center":
            delta_bending = load_value * L**3 / (48 * EI)
            delta_shear   = load_value * L / (4 * kGA)
            M_max = load_value * L / 4
            V_max = load_value / 2
        elif load_type == "uniform":
            delta_bending = 5 * load_value * L**4 / (384 * EI)
            delta_shear   = load_value * L**2 / (8 * kGA)
            M_max = load_value * L**2 / 8
            V_max = load_value * L / 2
        else:
            return {"status": "error", "message": f"Unsupported load_type '{load_type}' for simply_supported."}

    elif support_type == "cantilever":
        if load_type == "point_end":
            delta_bending = load_value * L**3 / (3 * EI)
            delta_shear   = load_value * L / kGA
            M_max = load_value * L
            V_max = load_value
        elif load_type == "uniform":
            delta_bending = load_value * L**4 / (8 * EI)
            delta_shear   = load_value * L**2 / (2 * kGA)
            M_max = load_value * L**2 / 2
            V_max = load_value * L
        else:
            return {"status": "error", "message": f"Unsupported load_type '{load_type}' for cantilever."}

    else:
        return {"status": "error", "message": f"Unknown support_type '{support_type}'."}

    delta_max = delta_bending + delta_shear
    shear_contribution = (delta_shear / delta_max * 100) if delta_max > 0 else 0.0

    d_approx = 2 * np.sqrt(I)
    slenderness = L / d_approx if d_approx > 0 else float("inf")
    slenderness_note = (
        f"Timoshenko recommended (L/d = {slenderness:.1f} <= 20)"
        if slenderness <= 20
        else f"NOTE: L/d = {slenderness:.1f} — Euler-Bernoulli may be sufficient"
    )

    return {
        "status": "ok",
        "theory": "Timoshenko",
        "support_type": support_type,
        "load_type": load_type,
        "EI_Nm2": round(EI, 4),
        "kGA_N": round(kGA, 4),
        "phi_shear_parameter": round(phi, 6),
        "delta_bending_m": round(delta_bending, 8),
        "delta_shear_m": round(delta_shear, 8),
        "delta_max_m": round(delta_max, 8),
        "shear_contribution_%": round(shear_contribution, 2),
        "M_max_Nm": round(M_max, 4),
        "V_max_N": round(V_max, 4),
        "slenderness_L_d": round(slenderness, 2),
        "slenderness_note": slenderness_note,
        "note": "phi > 0.1 means shear deformation is significant",
    }