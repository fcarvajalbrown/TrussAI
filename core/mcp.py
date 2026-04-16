from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

def get_mcp_client() -> MCPClient:
    """Return a configured MCP filesystem client pointing to ./configs."""
    return MCPClient(
        lambda: stdio_client(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "./configs"]
        ))
    )