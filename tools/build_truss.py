from strands import tool
import numpy as np

@tool
def build_truss(nodes: list, members: list, supports: list) -> dict:
    """
    Assemble the global stiffness matrix for a 2D truss.
    nodes: [[x, y], ...] coordinates in meters
    members: [[node_i, node_j, area, E], ...] area in m2, E in Pa
    supports: [[node_id, dof_x, dof_y], ...] 1=fixed, 0=free
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes  # 2 DOFs per node (x, y)
    K = np.zeros((n_dof, n_dof))

    member_data = []

    for member in members:
        i, j, A, E = int(member[0]), int(member[1]), member[2], member[3]
        xi, yi = nodes[i]
        xj, yj = nodes[j]

        # member length + direction cosines
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        c = (xj - xi) / L  # cos theta
        s = (yj - yi) / L  # sin theta

        # local stiffness matrix (4x4) via direct stiffness method
        k = (A * E / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s],
        ])

        # assemble into global K
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k[a, b]

        member_data.append({
            "i": i, "j": j,
            "L": round(L, 4),
            "A": A, "E": E,
            "c": round(c, 4),
            "s": round(s, 4),
        })

    # cache for solve_truss and analyze_results
    build_truss.K = K
    build_truss.nodes = nodes
    build_truss.members = member_data
    build_truss.supports = supports
    build_truss.n_dof = n_dof

    return {
        "status": "ok",
        "n_nodes": n_nodes,
        "n_members": len(members),
        "n_dof": n_dof,
        "members": member_data,
        "K_shape": list(K.shape),
    }