# tavily_server.py
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from sse import create_sse_server
from typing import Annotated, Literal
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, field_validator
from tavily import TavilyClient, InvalidAPIKeyError, UsageLimitExceededError
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
mcp = FastMCP("Tavily Search")

# Mount the Starlette‐based SSE server under "/"
app.mount("/", create_sse_server(mcp))

# Get API key from environment
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required")

# Initialize Tavily client
client = TavilyClient(api_key=api_key)

class SearchBase(BaseModel):
    """Base parameters for Tavily search."""
    query: Annotated[str, Field(description="Search query")]
    max_results: Annotated[
        int,
        Field(
            default=5,
            description="Maximum number of results to return",
            gt=0,
            lt=20,
        ),
    ]
    include_domains: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="List of domains to specifically include in the search results (e.g. ['example.com', 'test.org'] or 'example.com')",
        ),
    ]
    exclude_domains: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="List of domains to specifically exclude from the search results (e.g. ['example.com', 'test.org'] or 'example.com')",
        ),
    ]

    @field_validator('include_domains', 'exclude_domains', mode='before')
    @classmethod
    def parse_domains_list(cls, v):
        """Parse domain lists from various input formats.
        
        Handles:
        - None -> []
        - String JSON arrays -> list
        - Single domain string -> [string]
        - Comma-separated string -> list of domains
        - List of domains -> unchanged
        """
        if v is None:
            return []
        if isinstance(v, list):
            return [domain.strip() for domain in v if domain.strip()]
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            try:
                # Try to parse as JSON string
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [domain.strip() for domain in parsed if domain.strip()]
                return [parsed.strip()]  # Single value from JSON
            except json.JSONDecodeError:
                # Not JSON, check if comma-separated
                if ',' in v:
                    return [domain.strip() for domain in v.split(',') if domain.strip()]
                return [v]  # Single domain
        return []

class GeneralSearch(SearchBase):
    """Parameters for general web search."""
    search_depth: Annotated[
        Literal["basic", "advanced"],
        Field(
            default="basic",
            description="Depth of search - 'basic' or 'advanced'",
        ),
    ]

class AnswerSearch(SearchBase):
    """Parameters for search with answer."""
    search_depth: Annotated[
        Literal["basic", "advanced"],
        Field(
            default="advanced",
            description="Depth of search - 'basic' or 'advanced'",
        ),
    ]

class NewsSearch(SearchBase):
    """Parameters for news search."""
    days: Annotated[
        int | None,
        Field(
            default=None,
            description="Number of days back to search (default is 3)",
            gt=0,
            le=365,
        ),
    ]

def format_results(response: dict) -> str:
    """Format Tavily search results into a readable string."""
    output = []
    
    # Add domain filter information if present
    if response.get("included_domains") or response.get("excluded_domains"):
        filters = []
        if response.get("included_domains"):
            filters.append(f"Including domains: {', '.join(response['included_domains'])}")
        if response.get("excluded_domains"):
            filters.append(f"Excluding domains: {', '.join(response['excluded_domains'])}")
        output.append("Search Filters:")
        output.extend(filters)
        output.append("")  # Empty line for separation
    
    if response.get("answer"):
        output.append(f"Answer: {response['answer']}")
        output.append("\nSources:")
        # Add immediate source references for the answer
        for result in response["results"]:
            output.append(f"- {result['title']}: {result['url']}")
        output.append("")  # Empty line for separation
    
    output.append("Detailed Results:")
    for result in response["results"]:
        output.append(f"\nTitle: {result['title']}")
        output.append(f"URL: {result['url']}")
        output.append(f"Content: {result['content']}")
        if result.get("published_date"):
            output.append(f"Published: {result['published_date']}")
        
    return "\n".join(output)

@mcp.tool()
async def tavily_web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: str = "",
    exclude_domains: str = ""
) -> str:
    """Performs a comprehensive web search using Tavily's AI-powered search engine. 
    Excels at extracting and summarizing relevant content from web pages, making it ideal for research, 
    fact-finding, and gathering detailed information. Can run in either 'basic' mode for faster, simpler searches 
    or 'advanced' mode for more thorough analysis. Basic is cheaper and good for most use cases. 
    Supports filtering results by including or excluding specific domains.
    Use include_domains/exclude_domains parameters to filter by specific websites.
    Returns multiple search results with AI-extracted relevant content.

    Args:
        query: Search query
        max_results: Maximum number of results to return (1-19)
        search_depth: Depth of search - 'basic' or 'advanced'
        include_domains: Comma-separated list of domains to include (e.g., "example.com,test.org")
        exclude_domains: Comma-separated list of domains to exclude (e.g., "spam.com,ads.net")

    Returns:
        Formatted search results with titles, URLs, and content
    """
    try:
        # Validate search_depth parameter
        if search_depth not in ["basic", "advanced"]:
            search_depth = "basic"
        
        # Parse domain lists from strings
        include_list = []
        exclude_list = []
        
        if include_domains and include_domains.strip():
            include_list = [domain.strip() for domain in include_domains.split(',') if domain.strip()]
        
        if exclude_domains and exclude_domains.strip():
            exclude_list = [domain.strip() for domain in exclude_domains.split(',') if domain.strip()]
            
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_list,
            exclude_domains=exclude_list,
        )
        
        # Add domain filter information to response for formatting
        if include_list:
            response["included_domains"] = include_list
        if exclude_list:
            response["excluded_domains"] = exclude_list
            
        return format_results(response)
        
    except (InvalidAPIKeyError, UsageLimitExceededError) as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
    except ValueError as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

@mcp.tool()
async def tavily_answer_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: str = "",
    exclude_domains: str = ""
) -> str:
    """Performs a web search using Tavily's AI search engine and generates a direct answer to the query, 
    along with supporting search results. Best used for questions that need concrete answers backed by current web sources. 
    Uses advanced search depth by default for comprehensive analysis.

    Features powerful source control through domain filtering:
    - For academic research: exclude_domains="wikipedia.org" for more scholarly sources
    - For financial analysis: include_domains="wsj.com,bloomberg.com,ft.com"
    - For technical documentation: include_domains="docs.python.org,developer.mozilla.org"
    - For scientific papers: include_domains="nature.com,sciencedirect.com"
    - Can combine includes and excludes to fine-tune your sources

    Particularly effective for factual queries, technical questions, and queries requiring synthesis of multiple sources.

    Args:
        query: Search query
        max_results: Maximum number of results to return (1-19)
        search_depth: Depth of search - 'basic' or 'advanced'
        include_domains: Comma-separated list of domains to include (e.g., "wsj.com,bloomberg.com")
        exclude_domains: Comma-separated list of domains to exclude (e.g., "wikipedia.org,reddit.com")

    Returns:
        AI-generated answer with supporting search results
    """
    try:
        # Validate search_depth parameter
        if search_depth not in ["basic", "advanced"]:
            search_depth = "advanced"
        
        # Parse domain lists from strings
        include_list = []
        exclude_list = []
        
        if include_domains and include_domains.strip():
            include_list = [domain.strip() for domain in include_domains.split(',') if domain.strip()]
        
        if exclude_domains and exclude_domains.strip():
            exclude_list = [domain.strip() for domain in exclude_domains.split(',') if domain.strip()]
            
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
            include_domains=include_list,
            exclude_domains=exclude_list,
        )
        
        # Add domain filter information to response for formatting
        if include_list:
            response["included_domains"] = include_list
        if exclude_list:
            response["excluded_domains"] = exclude_list
            
        return format_results(response)
        
    except (InvalidAPIKeyError, UsageLimitExceededError) as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
    except ValueError as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

@mcp.tool()
async def tavily_news_search(
    query: str,
    max_results: int = 5,
    days: int = 3,
    include_domains: str = "",
    exclude_domains: str = ""
) -> str:
    """Searches recent news articles using Tavily's specialized news search functionality. 
    Ideal for current events, recent developments, and trending topics. Can filter results by recency 
    (number of days back to search) and by including or excluding specific news domains.

    Powerful domain filtering for news sources:
    - For mainstream news: include_domains="reuters.com,apnews.com,bbc.com"
    - For financial news: include_domains="bloomberg.com,wsj.com,ft.com"
    - For tech news: include_domains="techcrunch.com,theverge.com"
    - To exclude paywalled content: exclude_domains="wsj.com,ft.com"
    - To focus on specific regions: include_domains="bbc.co.uk" for UK news

    Returns news articles with publication dates and relevant excerpts.

    Args:
        query: Search query
        max_results: Maximum number of results to return (1-19)
        days: Number of days back to search (default is 3, max 365)
        include_domains: Comma-separated list of domains to include (e.g., "reuters.com,bbc.com")
        exclude_domains: Comma-separated list of domains to exclude (e.g., "tabloid.com,spam.net")

    Returns:
        Recent news articles with publication dates and excerpts
    """
    try:
        # Parse domain lists from strings
        include_list = []
        exclude_list = []
        
        if include_domains and include_domains.strip():
            include_list = [domain.strip() for domain in include_domains.split(',') if domain.strip()]
        
        if exclude_domains and exclude_domains.strip():
            exclude_list = [domain.strip() for domain in exclude_domains.split(',') if domain.strip()]
            
        response = client.search(
            query=query,
            max_results=max_results,
            topic="news",
            days=days,
            include_domains=include_list,
            exclude_domains=exclude_list,
        )
        
        # Add domain filter information to response for formatting
        if include_list:
            response["included_domains"] = include_list
        if exclude_list:
            response["excluded_domains"] = exclude_list
            
        return format_results(response)
        
    except (InvalidAPIKeyError, UsageLimitExceededError) as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
    except ValueError as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))