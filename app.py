import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import uuid
import json
from agent import get_agent, chat
from core.mcp import get_mcp_client

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrussAI",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e8e6f0;
    font-family: 'Syne', sans-serif;
}
.stApp { background-color: #0a0a0f; }

.truss-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    color: #e8e6f0;
    line-height: 1;
}
.truss-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #555570;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 20px;
}
.msg-user {
    background: #16161f;
    border-left: 2px solid #4a6cf7;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #b0aec8;
}
.msg-agent {
    background: #12121a;
    border-left: 2px solid #2dd4a7;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #e8e6f0;
    white-space: pre-wrap;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #444460;
    margin-bottom: 8px;
    border-bottom: 1px solid #1a1a28;
    padding-bottom: 6px;
}
section[data-testid="stSidebar"] { display: none; }

/* chip buttons */
div[data-testid="stButton"] button {
    background: #16161f !important;
    color: #8888aa !important;
    border: 1px solid #2a2a3a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    border-radius: 20px !important;
    padding: 4px 12px !important;
    margin: 2px !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #4a6cf7 !important;
    color: #4a6cf7 !important;
}

/* chat input */
div[data-testid="stChatInput"] textarea {
    background-color: #12121a !important;
    border: 1px solid #2a2a3a !important;
    color: #e8e6f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "build_result" not in st.session_state:
    st.session_state.build_result = None
if "solve_result" not in st.session_state:
    st.session_state.solve_result = None
if "auto_send" not in st.session_state:
    st.session_state.auto_send = None
if "agent" not in st.session_state:
    mcp = get_mcp_client()
    mcp.__enter__()
    st.session_state.mcp = mcp
    st.session_state.agent = get_agent(mcp, f"trussai-{uuid.uuid4().hex[:8]}")


# ── helpers ───────────────────────────────────────────────────────────────────

def send_message(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("thinking…"):
        response = chat(st.session_state.agent, prompt)
    st.session_state.messages.append({"role": "agent", "content": response})


def draw_truss(build_result, solve_result=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    ax.tick_params(colors="#444460")
    for spine in ax.spines.values():
        spine.set_color("#1a1a28")

    nodes = np.array(build_result["nodes"])
    members = build_result["members"]
    scale = 500

    for m in members:
        i, j = m["i"], m["j"]
        ax.plot([nodes[i][0], nodes[j][0]],
                [nodes[i][1], nodes[j][1]],
                color="#4a6cf7", linewidth=2, zorder=1)

    if solve_result and solve_result.get("status") == "ok":
        u = np.array(solve_result["u"])
        for m in members:
            i, j = m["i"], m["j"]
            ax.plot([nodes[i][0] + scale*u[2*i],   nodes[j][0] + scale*u[2*j]],
                    [nodes[i][1] + scale*u[2*i+1], nodes[j][1] + scale*u[2*j+1]],
                    color="#2dd4a7", linewidth=1.5, linestyle="--", zorder=2, alpha=0.8)

    ax.scatter(nodes[:, 0], nodes[:, 1], color="#e8e6f0", s=50, zorder=3)
    for idx, (x, y) in enumerate(nodes):
        ax.annotate(f" {idx}", (x, y), color="#8888aa",
                    fontsize=8, fontfamily="monospace")

    handles = [mpatches.Patch(color="#4a6cf7", label="Original")]
    if solve_result and solve_result.get("status") == "ok":
        handles.append(mpatches.Patch(color="#2dd4a7", label=f"Deformed ×{scale}"))
    ax.legend(handles=handles, facecolor="#12121a", labelcolor="#8888aa",
              fontsize=7, framealpha=0.8, edgecolor="#2a2a3a")
    ax.set_title("TRUSS DIAGRAM", color="#444460",
                 fontsize=8, fontfamily="monospace", loc="left", pad=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


# ── auto-send from prompt chips ───────────────────────────────────────────────

if st.session_state.auto_send:
    prompt = st.session_state.auto_send
    st.session_state.auto_send = None
    send_message(prompt)

# ── layout ────────────────────────────────────────────────────────────────────

col_viz, col_chat = st.columns([1.3, 1])

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_viz:
    st.markdown('<div class="truss-header">TrussAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="truss-sub">Structural analysis · AWS Strands · Ollama qwen3.5:2b</div>',
                unsafe_allow_html=True)

    if st.session_state.build_result:
        fig = draw_truss(st.session_state.build_result, st.session_state.solve_result)
        st.pyplot(fig)
        plt.close()
    else:
        st.markdown("""
        <div style="height:300px;border:1px dashed #2a2a3a;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    color:#333350;font-family:monospace;font-size:0.75rem;text-align:center;line-height:2;">
            no truss loaded<br>ask the agent to build one
        </div>""", unsafe_allow_html=True)

    if st.session_state.solve_result and \
       st.session_state.solve_result.get("status") == "ok":
        st.markdown('<div class="section-label">Displacements</div>', unsafe_allow_html=True)
        for d in st.session_state.solve_result["displacements"]:
            st.markdown(
                f'<div class="msg-user">Node {d["node"]} &nbsp;|&nbsp; '
                f'ux: {d["ux"]:.6f} m &nbsp;|&nbsp; uy: {d["uy"]:.6f} m</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Load results for visualization</div>',
                unsafe_allow_html=True)

    with st.expander("Paste build_result JSON"):
        raw = st.text_area("build_result", height=80, label_visibility="collapsed",
                           placeholder='{"status":"ok","nodes":...}')
        if st.button("Load build", key="load_build"):
            try:
                st.session_state.build_result = json.loads(raw)
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    with st.expander("Paste solve_result JSON"):
        raw2 = st.text_area("solve_result", height=80, label_visibility="collapsed",
                            placeholder='{"status":"ok","u":...}')
        if st.button("Load solve", key="load_solve"):
            try:
                st.session_state.solve_result = json.loads(raw2)
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_chat:
    st.markdown('<div class="section-label">Quick prompts</div>', unsafe_allow_html=True)

    PROMPTS = [
        "Build a 3-node triangle truss and solve with 10kN load at apex",
        "Apply a 10kN downward load at node 2 and solve",
        "Is this truss safe for structural steel?",
        "Explain the math behind the stiffness matrix",
        "Analyze a simply supported beam using Euler-Bernoulli theory",
        "Which beam theory applies to a short thick beam?",
    ]

    cols = st.columns(2)
    for idx, prompt in enumerate(PROMPTS):
        label = prompt[:38] + "…" if len(prompt) > 38 else prompt
        with cols[idx % 2]:
            if st.button(label, key=f"chip_{idx}"):
                st.session_state.auto_send = prompt
                st.rerun()

    st.markdown('<div class="section-label" style="margin-top:16px;">Agent Chat</div>',
                unsafe_allow_html=True)

    # chat history
    chat_container = st.container(height=380)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                '<div style="color:#333350;font-family:monospace;font-size:0.75rem;'
                'padding:20px 0;">no messages yet — type below or click a prompt</div>',
                unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user">▸ {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-agent">{msg["content"]}</div>',
                            unsafe_allow_html=True)

    # native chat input — always visible, Enter to send
    if prompt := st.chat_input("describe your structure or ask a question…"):
        send_message(prompt)
        st.rerun()

    if st.button("Clear chat", key="clear_btn"):
        st.session_state.messages = []
        st.session_state.build_result = None
        st.session_state.solve_result = None
        st.rerun()
