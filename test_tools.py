"""
TrussAI tool test suite.
Run with: pytest test_tools.py -v
Ollama integration test: pytest test_tools.py -v -m integration
"""
import pytest
import numpy as np
from tools.truss import build_truss, solve_truss, analyze_results
from tools.beam import euler_beam, timoshenko_beam


# ── fixtures ──────────────────────────────────────────────────────────────────

NODES    = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
MEMBERS  = [[0, 1, 0.01, 200e9], [1, 2, 0.01, 200e9]]
SUPPORTS = [[0, 1, 1], [2, 1, 1]]
LOADS    = [[1, 0.0, -10000.0]]


# ── build_truss ───────────────────────────────────────────────────────────────

def test_build_truss_returns_ok():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    assert result["status"] == "ok"

def test_build_truss_node_count():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    assert result["n_nodes"] == 3

def test_build_truss_member_count():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    assert result["n_members"] == 2

def test_build_truss_dof_count():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    assert result["n_dof"] == 6

def test_build_truss_K_shape():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    K = np.array(result["K"])
    assert K.shape == (6, 6)

def test_build_truss_K_symmetric():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    K = np.array(result["K"])
    assert np.allclose(K, K.T)

def test_build_truss_returns_members():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    assert len(result["members"]) == 2

def test_build_truss_returns_nodes():
    result = build_truss(NODES, MEMBERS, SUPPORTS)
    assert result["nodes"] == NODES


# ── solve_truss ───────────────────────────────────────────────────────────────

def test_solve_truss_invalid_build_returns_error():
    result = solve_truss({"status": "error"}, LOADS)
    assert result["status"] == "error"

def test_solve_truss_returns_ok():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    assert result["status"] == "ok"

def test_solve_truss_displacement_count():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    assert len(result["displacements"]) == 3

def test_solve_truss_supported_nodes_zero_displacement():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    u = result["u"]
    assert abs(u[1]) < 1e-10  # node 0 uy
    assert abs(u[5]) < 1e-10  # node 2 uy

def test_solve_truss_loaded_node_deflects_down():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    assert result["u"][3] < 0  # node 1 uy negative

def test_solve_truss_reaction_forces_balance():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    vertical = [r["force_N"] for r in result["reactions"] if r["dof"] % 2 == 1]
    assert abs(sum(vertical) + 10000.0) < 1e-4

def test_solve_truss_returns_u():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    assert "u" in result
    assert len(result["u"]) == 6

def test_solve_truss_passes_members_forward():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    result = solve_truss(build, LOADS)
    assert "members" in result
    assert len(result["members"]) == 2


# ── analyze_results ───────────────────────────────────────────────────────────

def test_analyze_invalid_solve_returns_error():
    result = analyze_results({"status": "error"})
    assert result["status"] == "error"

def test_analyze_returns_ok():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    solve = solve_truss(build, LOADS)
    result = analyze_results(solve)
    assert result["status"] == "ok"

def test_analyze_member_count():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    solve = solve_truss(build, LOADS)
    result = analyze_results(solve)
    assert len(result["members"]) == 2

def test_analyze_safe_flag_is_bool():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    solve = solve_truss(build, LOADS)
    result = analyze_results(solve, yield_strength=250e6)
    assert isinstance(result["safe"], bool)

def test_analyze_stress_values_present():
    build = build_truss(NODES, MEMBERS, SUPPORTS)
    solve = solve_truss(build, LOADS)
    result = analyze_results(solve)
    for m in result["members"]:
        assert "stress_MPa" in m
        assert "utilization_%" in m


# ── euler_beam ────────────────────────────────────────────────────────────────

def test_euler_simply_supported_point_center():
    result = euler_beam(
        length=5.0, E=200e9, I=1e-4,
        load_type="point_center", load_value=10000.0,
        support_type="simply_supported"
    )
    assert result["status"] == "ok"
    expected = 10000 * 5**3 / (48 * 200e9 * 1e-4)
    assert abs(result["delta_max_m"] - expected) < 1e-6

def test_euler_cantilever_point_end():
    result = euler_beam(
        length=2.0, E=200e9, I=1e-4,
        load_type="point_end", load_value=5000.0,
        support_type="cantilever"
    )
    assert result["status"] == "ok"
    expected = 5000 * 2**3 / (3 * 200e9 * 1e-4)
    assert abs(result["delta_max_m"] - expected) < 1e-6

def test_euler_fixed_fixed_uniform():
    result = euler_beam(
        length=4.0, E=200e9, I=1e-4,
        load_type="uniform", load_value=2000.0,
        support_type="fixed_fixed"
    )
    assert result["status"] == "ok"
    expected = 2000 * 4**4 / (384 * 200e9 * 1e-4)
    assert abs(result["delta_max_m"] - expected) < 1e-6

def test_euler_slenderness_warning():
    result = euler_beam(
        length=0.1, E=200e9, I=1e-4,
        load_type="point_center", load_value=1000.0,
        support_type="simply_supported"
    )
    assert "WARNING" in result["slenderness_note"]

def test_euler_unknown_support_returns_error():
    result = euler_beam(
        length=5.0, E=200e9, I=1e-4,
        load_type="uniform", load_value=1000.0,
        support_type="unknown"
    )
    assert result["status"] == "error"


# ── timoshenko_beam ───────────────────────────────────────────────────────────

def test_timoshenko_simply_supported_point_center():
    result = timoshenko_beam(
        length=1.0, E=200e9, I=1e-4,
        G=80e9, A=0.01, kappa=0.833,
        load_type="point_center", load_value=10000.0,
        support_type="simply_supported"
    )
    assert result["status"] == "ok"

def test_timoshenko_deflection_greater_than_euler():
    L, E, I, P = 1.0, 200e9, 1e-4, 10000.0
    euler = euler_beam(length=L, E=E, I=I, load_type="point_center",
                       load_value=P, support_type="simply_supported")
    timo = timoshenko_beam(length=L, E=E, I=I, G=80e9, A=0.01, kappa=0.833,
                           load_type="point_center", load_value=P,
                           support_type="simply_supported")
    assert timo["delta_max_m"] > euler["delta_max_m"]

def test_timoshenko_shear_contribution_positive():
    result = timoshenko_beam(
        length=1.0, E=200e9, I=1e-4,
        G=80e9, A=0.01, kappa=0.833,
        load_type="uniform", load_value=5000.0,
        support_type="cantilever"
    )
    assert result["shear_contribution_%"] > 0

def test_timoshenko_phi_present():
    result = timoshenko_beam(
        length=2.0, E=200e9, I=1e-4,
        G=80e9, A=0.01, kappa=0.833,
        load_type="point_end", load_value=1000.0,
        support_type="cantilever"
    )
    assert "phi_shear_parameter" in result

def test_timoshenko_unknown_support_returns_error():
    result = timoshenko_beam(
        length=2.0, E=200e9, I=1e-4,
        G=80e9, A=0.01, kappa=0.833,
        load_type="uniform", load_value=1000.0,
        support_type="fixed_fixed"
    )
    assert result["status"] == "error"


# ── ollama integration ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_agent_responds_to_structural_question():
    """
    Requires: ollama serve running + qwen2.5:3b pulled.
    Run with: pytest test_tools.py -v -m integration
    """
    from agent import get_agent, chat
    from core.mcp import get_mcp_client

    mcp_client = get_mcp_client()
    with mcp_client:
        agent = get_agent(mcp_client)
        response = chat(agent, "What is the Euler-Bernoulli beam theory used for?")

    assert isinstance(response, str)
    assert len(response) > 20