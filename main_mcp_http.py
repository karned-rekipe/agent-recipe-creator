from pathlib import Path

from adapters.input.fastmcp.register import register_tools
from arclith import Arclith
from infrastructure.logging_setup import setup_logging

_logger = setup_logging()

arclith = Arclith(Path(__file__).parent / "config.yaml")
_logger.info("🚀 MCP HTTP server starting", host = arclith.config.mcp.host, port = arclith.config.mcp.port)
mcp = arclith.fastmcp("Rekipe - Recipe Creator Agent")
register_tools(mcp, arclith)

if __name__ == "__main__":
    arclith.run_mcp_http(mcp)
