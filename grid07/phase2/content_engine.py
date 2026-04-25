"""
Phase 2: Autonomous Content Engine — LangGraph State Machine
Flow: decide_search → web_search → draft_post → structured JSON output
"""

import json
import os
import re
from typing import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# LLM — swap to ChatOpenAI / ChatGroq by changing this one import
from langchain_ollama import ChatOllama           # pip install langchain-ollama
# from langchain_openai import ChatOpenAI
# from langchain_groq   import ChatGroq

# ──────────────────────────────────────────────
# Shared LLM instance
# ──────────────────────────────────────────────
LLM = ChatOllama(model="llama3", temperature=0.7)
# LLM = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
# LLM = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))


# ──────────────────────────────────────────────
# Mock search tool (simulates SearXNG results)
# ──────────────────────────────────────────────
MOCK_NEWS: dict[str, list[str]] = {
    "crypto":     ["Bitcoin hits new all-time high amid regulatory ETF approvals",
                   "Ethereum layer-2 solutions cut gas fees by 90%"],
    "ai":         ["OpenAI GPT-5 passes bar exam with 97% score",
                   "Anthropic raises $2B for safety-focused AI research"],
    "market":     ["S&P 500 surges on strong jobs report; Fed hints at rate pause",
                   "Nvidia market cap crosses $3 trillion — analysts divided"],
    "regulation": ["EU AI Act enforcement begins — Big Tech faces billion-euro fines",
                   "US Senate debates sweeping data-privacy legislation"],
    "climate":    ["Record heatwaves force hospitals to ration power",
                   "Solar energy now cheaper than coal in 90% of the world"],
    "space":      ["SpaceX Starship successfully completes full orbital test",
                   "NASA Artemis crew lands on lunar south pole"],
}

@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search. Returns hardcoded headlines
    relevant to the keywords found in *query*.
    """
    query_lower = query.lower()
    results: list[str] = []

    for keyword, headlines in MOCK_NEWS.items():
        if keyword in query_lower:
            results.extend(headlines)

    if not results:
        results = ["No major breaking news found — consider a trending angle."]

    formatted = "\n".join(f"• {h}" for h in results[:4])
    print(f"  [mock_searxng_search] query='{query}'\n  Results:\n{formatted}")
    return formatted


# ──────────────────────────────────────────────
# LangGraph state schema
# ──────────────────────────────────────────────
class GraphState(TypedDict):
    bot_id:       str
    persona:      str
    search_query: str
    search_results: str
    post_json:    dict   # final structured output


# ──────────────────────────────────────────────
# Node 1 — Decide what to post about / build search query
# ──────────────────────────────────────────────
def node_decide_search(state: GraphState) -> GraphState:
    print("\n[Node 1] Deciding topic & search query …")

    system_prompt = (
        "You are the following persona:\n"
        f"{state['persona']}\n\n"
        "Based on your persona, decide ONE topic you want to post about today. "
        "Respond with ONLY a short web-search query (5-10 words, no punctuation)."
    )

    response = LLM.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content="What do you want to search for today?")])

    query = response.content.strip().strip('"').strip("'")
    print(f"  → Search query: '{query}'")
    return {**state, "search_query": query}


# ──────────────────────────────────────────────
# Node 2 — Execute mock web search
# ──────────────────────────────────────────────
def node_web_search(state: GraphState) -> GraphState:
    print("\n[Node 2] Running web search …")
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    return {**state, "search_results": results}


# ──────────────────────────────────────────────
# Node 3 — Draft a 280-char opinionated post
# ──────────────────────────────────────────────
def node_draft_post(state: GraphState) -> GraphState:
    print("\n[Node 3] Drafting post …")

    system_prompt = (
        "You are the following persona — stay fully in character:\n"
        f"{state['persona']}\n\n"
        "You have just read these real-world headlines:\n"
        f"{state['search_results']}\n\n"
        "Write a SINGLE opinionated social-media post (max 280 characters). "
        "Then return ONLY a valid JSON object with these exact keys and NO extra text:\n"
        '{"bot_id": "<bot_id>", "topic": "<one-word topic>", "post_content": "<your post>"}\n'
        "No markdown, no code fences, no explanation — raw JSON only."
    )

    user_msg = f"Write the post for bot_id='{state['bot_id']}'."

    response = LLM.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content=user_msg)])

    raw = response.content.strip()

    # Strip markdown code fences if the model added them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        post_json = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback — extract JSON block if buried in prose
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        post_json = json.loads(match.group()) if match else {
            "bot_id": state["bot_id"],
            "topic": "unknown",
            "post_content": raw[:280],
        }

    print(f"  → Structured output: {json.dumps(post_json, indent=2)}")
    return {**state, "post_json": post_json}


# ──────────────────────────────────────────────
# Build & compile the graph
# ──────────────────────────────────────────────
def build_content_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


# ──────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────
def generate_post(bot_id: str, persona: str) -> dict:
    """Run the full LangGraph pipeline for a single bot and return the JSON post."""
    app = build_content_graph()
    initial_state: GraphState = {
        "bot_id":         bot_id,
        "persona":        persona,
        "search_query":   "",
        "search_results": "",
        "post_json":      {},
    }
    final_state = app.invoke(initial_state)
    return final_state["post_json"]


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from phase1.router import BOT_PERSONAS

    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\n{'='*60}")
        print(f"🤖 Running content engine for {bot_id}")
        result = generate_post(bot_id, persona)
        print(f"\n📝 Final post JSON:\n{json.dumps(result, indent=2)}")
