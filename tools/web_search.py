from langchain_community.tools import DuckDuckGoSearchResults

# Sử dụng phiên bản ổn định hơn
web_search_tool = DuckDuckGoSearchResults(
    name="web_search",
    description="Search the web for current information. Returns top results.",
    num_results=3
)
