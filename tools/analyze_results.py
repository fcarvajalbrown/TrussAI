from strands import tool
import numpy as np
from tools.build_truss import build_truss
from tools.solve_truss import solve_truss

@tool
def analyze_results(yield_strength: float = 250e6) -> dict:
    """
    Compute member stresses and check against yield strength.
    yield_strength: material yield strength in Pa (default 250 MPa for structural steel)
    """
    u = getattr(solve_truss, "u", None)
    if u is None:
        return {"status": "error", "message": "No solution found — run solve_truss first."}

    members = build_truss.members
    results = []
    warnings = []
    failures = []

    for m in members:
        i, j = m["i"], m["j"]
        L, A, E = m["L"], m["A"], m["E"]
        c, s = m["c"], m["s"]

        # member elongation from nodal displacements
        ui = [u[2*i], u[2*i+1]]
        uj = [u[2*j], u[2*j+1]]
        delta = c * (uj[0] - ui[0]) + s * (uj[1] - ui[1])

        # axial strain + stress
        strain = delta / L
        stress = E * strain
        utilization = abs(stress) / yield_strength * 100

        status = "ok"
        if utilization > 100:
            status = "FAILURE"
            failures.append({
                "member": f"{i}-{j}",
                "stress_MPa": round(stress / 1e6, 2),
            })
        elif utilization > 80:
            status = "WARNING"
            warnings.append({
                "member": f"{i}-{j}",
                "utilization_%": round(utilization, 1),
            })

        results.append({
            "member": f"{i}-{j}",
            "stress_MPa": round(stress / 1e6, 4),
            "strain": round(strain, 8),
            "utilization_%": round(utilization, 2),
            "status": status,
        })

    # cache for Streamlit color coding
    analyze_results.last_results = results

    return {
        "status": "ok",
        "members": results,
        "warnings": warnings,
        "failures": failures,
        "safe": len(failures) == 0,
        "yield_strength_MPa": yield_strength / 1e6,
    }