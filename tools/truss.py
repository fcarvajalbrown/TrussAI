from strands import tool
import numpy as np


@tool
def build_truss(nodes: list, members: list, supports: list) -> dict:
    """
    Assemble the global stiffness matrix for a 2D truss.
    nodes: [[x, y], ...] coordinates in meters
    members: [[node_i, node_j, area, E], ...] area in m2, E in Pa
    supports: [[node_id, dof_x, dof_y], ...] 1=fixed, 0=free
    Returns stiffness matrix data and member geometry for use in solve_truss.
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))
    member_data = []

    for member in members:
        i, j, A, E = int(member[0]), int(member[1]), member[2], member[3]
        xi, yi = nodes[i]
        xj, yj = nodes[j]

        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        cos = (xj - xi) / L
        sin = (yj - yi) / L

        k = (A * E / L) * np.array([
            [ cos*cos,  cos*sin, -cos*cos, -cos*sin],
            [ cos*sin,  sin*sin, -cos*sin, -sin*sin],
            [-cos*cos, -cos*sin,  cos*cos,  cos*sin],
            [-cos*sin, -sin*sin,  cos*sin,  sin*sin],
        ])

        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k[a, b]

        member_data.append({
            "i": i, "j": j,
            "L": round(L, 4),
            "A": A, "E": E,
            "c": round(cos, 4),
            "s": round(sin, 4),
        })

    return {
        "status": "ok",
        "n_nodes": n_nodes,
        "n_members": len(members),
        "n_dof": n_dof,
        "nodes": nodes,
        "members": member_data,
        "supports": supports,
        "K": K.tolist(),
    }


@tool
def solve_truss(build_result: dict, loads: list) -> dict:
    """
    Solve Ku=F for the assembled truss.
    build_result: full output dict from build_truss
    loads: [[node_id, fx, fy], ...] applied forces in Newtons
    Returns displacements, reactions, and member data for use in analyze_results.
    """
    if build_result.get("status") != "ok":
        return {"status": "error", "message": "Invalid build_result — run build_truss first."}

    K = np.array(build_result["K"])
    n_dof = build_result["n_dof"]
    supports = build_result["supports"]
    nodes = build_result["nodes"]

    F = np.zeros(n_dof)
    for load in loads:
        node_id, fx, fy = int(load[0]), load[1], load[2]
        F[2 * node_id]     += fx
        F[2 * node_id + 1] += fy

    constrained = []
    for support in supports:
        node_id, dof_x, dof_y = int(support[0]), support[1], support[2]
        if dof_x == 1:
            constrained.append(2 * node_id)
        if dof_y == 1:
            constrained.append(2 * node_id + 1)

    free = [d for d in range(n_dof) if d not in constrained]
    K_free = K[np.ix_(free, free)]
    F_free = F[free]

    try:
        u_free = np.linalg.solve(K_free, F_free)
    except np.linalg.LinAlgError:
        return {"status": "error", "message": "Singular stiffness matrix — check supports and connectivity."}

    u = np.zeros(n_dof)
    for idx, dof in enumerate(free):
        u[dof] = u_free[idx]

    reactions = K @ u - F

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
        "u": u.tolist(),
        "members": build_result["members"],
    }


@tool
def analyze_results(solve_result: dict, yield_strength: float = 250e6) -> dict:
    """
    Compute member stresses and check against yield strength.
    solve_result: full output dict from solve_truss
    yield_strength: material yield strength in Pa (default 250 MPa for structural steel)
    """
    if solve_result.get("status") != "ok":
        return {"status": "error", "message": "Invalid solve_result — run solve_truss first."}

    u = np.array(solve_result["u"])
    members = solve_result["members"]
    results = []
    warnings = []
    failures = []

    for m in members:
        i, j = m["i"], m["j"]
        L, A, E = m["L"], m["A"], m["E"]
        c, s = m["c"], m["s"]

        delta = c * (u[2*j] - u[2*i]) + s * (u[2*j+1] - u[2*i+1])
        strain = delta / L
        stress = E * strain
        utilization = abs(stress) / yield_strength * 100

        status = "ok"
        if utilization > 100:
            status = "FAILURE"
            failures.append({"member": f"{i}-{j}", "stress_MPa": round(stress / 1e6, 2)})
        elif utilization > 80:
            status = "WARNING"
            warnings.append({"member": f"{i}-{j}", "utilization_%": round(utilization, 1)})

        results.append({
            "member": f"{i}-{j}",
            "stress_MPa": round(stress / 1e6, 4),
            "strain": round(strain, 8),
            "utilization_%": round(utilization, 2),
            "status": status,
        })

    return {
        "status": "ok",
        "members": results,
        "warnings": warnings,
        "failures": failures,
        "safe": len(failures) == 0,
        "yield_strength_MPa": yield_strength / 1e6,
    }