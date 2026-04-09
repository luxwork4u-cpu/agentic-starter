from langchain_community.tools import DuckDuckGoSearchRun

web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current information. Returns top results."
)
