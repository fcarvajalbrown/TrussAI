from strands import Agent
from strands.session.file_session_manager import FileSessionManager
from core.model import model
from core.mcp import get_mcp_client
from tools.truss import build_truss, solve_truss, analyze_results
from tools.beam import euler_beam, timoshenko_beam

session_manager = FileSessionManager(
    session_id="trussai-session",
    storage_dir="./sessions"
)

def get_agent(mcp_client, agent_id: str = "trussai") -> Agent:
    """Initialize agent with MCP tools + custom tools + session manager."""
    return Agent(
        model=model,
        system_prompt=open("prompts/system.md").read(),
        tools=[
            build_truss,
            solve_truss,
            analyze_results,
            euler_beam,
            timoshenko_beam,
        ] + mcp_client.list_tools_sync(),
        session_manager=session_manager,
        agent_id=agent_id,
    )

def chat(agent, user_input: str) -> str:
    """Send message to agent, return string response for Streamlit."""
    return str(agent(user_input))
