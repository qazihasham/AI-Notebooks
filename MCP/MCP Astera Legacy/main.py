# main.py
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from sse import create_sse_server
import math  # For square root calculation

app = FastAPI()
mcp = FastMCP("Calculator")

# Mount the Starlette‐based SSE server under “/”
app.mount("/", create_sse_server(mcp))

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of the two numbers.
    """
    return int(a) + int(b)

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The subtraction of the two numbers.
    """
    return int(a) - int(b)

@mcp.tool()
async def square_root(a: float) -> float:
    """Square Root a number.

    Args:
        a: The number to be square root.

    Returns:
        The square root of a number.
    """
    return math.sqrt(float(a))  # Fixed: Actually calculate square root