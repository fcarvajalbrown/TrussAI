from strands import tool
import numpy as np
from tools.build_truss import build_truss

@tool
def solve_truss(loads: list) -> dict:
    """
    Solve Ku=F for the assembled truss.
    loads: [[node_id, fx, fy], ...] applied forces in Newtons
    Returns nodal displacements and reaction forces.
    """
    K = getattr(build_truss, "K", None)
    if K is None:
        return {"status": "error", "message": "No truss built — run build_truss first."}

    nodes   = build_truss.nodes
    supports = build_truss.supports
    n_dof   = build_truss.n_dof

    # assemble global force vector
    F = np.zeros(n_dof)
    for load in loads:
        node_id, fx, fy = int(load[0]), load[1], load[2]
        F[2 * node_id]     += fx
        F[2 * node_id + 1] += fy

    # identify constrained DOFs
    constrained = []
    for support in supports:
        node_id, dof_x, dof_y = int(support[0]), support[1], support[2]
        if dof_x == 1:
            constrained.append(2 * node_id)
        if dof_y == 1:
            constrained.append(2 * node_id + 1)

    # free DOFs by exclusion
    free = [d for d in range(n_dof) if d not in constrained]

    # partition and solve reduced system
    K_free = K[np.ix_(free, free)]
    F_free = F[free]

    try:
        u_free = np.linalg.solve(K_free, F_free)
    except np.linalg.LinAlgError:
        return {"status": "error", "message": "Singular stiffness matrix — check supports and connectivity."}

    # rebuild full displacement vector
    u = np.zeros(n_dof)
    for idx, dof in enumerate(free):
        u[dof] = u_free[idx]

    # reaction forces at constrained DOFs
    reactions = K @ u - F

    # cache for analyze_results
    solve_truss.u = u
    solve_truss.F = F
    solve_truss.reactions = reactions

    displacements = [
        {"node": i, "ux": round(u[2*i], 8), "uy": round(u[2*i+1], 8)}
        for i in range(len(nodes))
    ]
    reaction_list = [
        {"dof": d, "force_N": round(float(reactions[d]), 4)}
        for d in constrained
    ]

    return {
        "status": "ok",
        "displacements": displacements,
        "reactions": reaction_list,
        "max_displacement_m": round(float(np.max(np.abs(u))), 8),
    }