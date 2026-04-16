import os
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.session.file_session_manager import FileSessionManager
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from tools.build_truss import build_truss
from tools.solve_truss import solve_truss
from tools.analyze_results import analyze_results
from tools.euler_beam import euler_beam
from tools.timoshenko_beam import timoshenko_beam

load_dotenv()

model = OllamaModel(
    model_id="qwen2.5:3b",
    host="http://localhost:11434",
)

session_manager = FileSessionManager(
    session_id="trussai-session",
    storage_dir="./sessions"
)

def get_mcp_client():
    """Return configured MCP filesystem client."""
    return MCPClient(
        lambda: stdio_client(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "./configs"]
        ))
    )

def get_agent(mcp_client):
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
        agent_id="trussai",
    )

def chat(agent, user_input: str) -> str:
    """Send message to agent, return string response for Streamlit."""
    return str(agent(user_input))