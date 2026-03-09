"""
JOEY MCP Server
===============
Exposes JOEY's vector search (pgvector + BGE embeddings) as an MCP-over-HTTP
server so that the GitHub Copilot SDK inside JOE (copilot_bridge) can call it
as a registered MCP tool.

How the Copilot SDK calls this server:
  GET  /mcp/tools          → Copilot discovers available tools on session start
  POST /mcp/tools/call     → Copilot calls a tool during a conversation turn

Two integration modes are documented in mcp-config.ts (copilot_bridge side).
This server handles BOTH — the difference is purely in how JOE registers it:

  MODE A — "Copilot Decides" (tool-based / autonomous)
  -------------------------------------------------------
    Registered as a normal MCP tool. Copilot reads the tool description and
    calls vector_search() only when it judges the user's question needs it.
    Best for: general assistant usage where not every message needs RAG.

  MODE B — "Always Search" (forced RAG / default)
  -------------------------------------------------------
    JOE's sessionController calls this server BEFORE sending the user's
    message to Copilot, injects the top results as a system context block,
    and then sends the enriched prompt. Copilot always has RAG context.
    Best for: engineering Q&A where the vector DB is the source of truth.

Run this server:
    cd joey_mcp_server
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Dependencies:
    pip install fastapi uvicorn psycopg2-binary sentence-transformers python-dotenv
"""

import sys
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Path setup ───────────────────────────────────────────────────────────────
# Allow importing from the sibling vector_db_manager package.
# joey_mcp_server/ and vector_db_manager/ sit in the same parent folder.
PARENT_DIR = Path(__file__).parent.parent
VECTOR_DB_DIR = PARENT_DIR / "JOE_MANAGER"
sys.path.insert(0, str(VECTOR_DB_DIR))

from embeddings.embedder import Embedder
from db.connection import DatabaseManager

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("joey_mcp_server")

# ─── App init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JOEY MCP Server",
    description="Vector search MCP bridge for JOE (Jabil Oracle for Engineering)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Singletons ───────────────────────────────────────────────────────────────
# Load embedding model once on startup. Reused for all requests.
# BGE model is already cached locally by vector_db_manager — no download needed.
embedder: Embedder = None


@app.on_event("startup")
async def startup():
    global embedder
    logger.info("Loading BGE embedding model...")
    embedder = Embedder()
    embedder.load()
    logger.info(f"Embedding model ready: {embedder.get_model_name()} ({embedder.get_dimensions()}d)")

    # Verify DB connectivity on startup so we fail fast if creds are wrong
    try:
        with DatabaseManager() as db:
            stats = db.get_stats()
            logger.info(
                f"Vector DB connected — "
                f"{stats['total_sources']} sources, "
                f"{stats['total_chunks']} chunks"
            )
    except Exception as e:
        logger.error(f"Vector DB connection failed on startup: {e}")
        # Don't crash — DB might come up later. Tool calls will return errors.


# ─── MCP Protocol Models ──────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    """
    Body sent by the Copilot SDK when calling a tool.
    The SDK sends: { "name": "vector_search", "parameters": { "query": "...", "top_k": 5 } }
    """
    name: str
    parameters: dict[str, Any] = {}


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


# ─── Tool Definitions ─────────────────────────────────────────────────────────
# These are returned by GET /mcp/tools so Copilot knows what tools exist
# and what parameters they expect. The description is critical — it's what
# Copilot reads to decide WHEN to call the tool (MODE A).

TOOL_DEFINITIONS = [
    {
        "name": "vector_search",
        "description": (
            "Semantic search over Jabil engineering documents (Work Instructions, SOPs, "
            "Visual Aids, PDFs, Excel sheets) stored in the vector database. "
            "Use this tool when the user asks about procedures, processes, standards, "
            "specifications, equipment setup, or any engineering topic that may be "
            "documented internally. Returns the most relevant text chunks with source "
            "file names and similarity scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language search query. Be specific — e.g. 'reflow oven temperature profile for lead-free' rather than just 'temperature'."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10).",
                    "default": 5
                },
                "include_context": {
                    "type": "boolean",
                    "description": "If true, also returns the neighboring chunks (before/after) for richer context. Default: true.",
                    "default": True
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_sources",
        "description": (
            "List all documents that have been ingested into the vector database. "
            "Use this when the user asks what documents are available, what files have "
            "been indexed, or wants to know if a specific document exists in the system."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# ─── MCP Endpoints ────────────────────────────────────────────────────────────

@app.get("/mcp/tools")
async def get_tools():
    """
    MCP tool discovery endpoint.
    Called by Copilot SDK on session start to register available tools.
    Returns list of tool definitions with names, descriptions, and parameter schemas.
    """
    return {"tools": TOOL_DEFINITIONS}


@app.post("/mcp/tools/call")
async def call_tool(request: ToolCallRequest):
    """
    MCP tool execution endpoint.
    Called by Copilot SDK when it decides to invoke a tool during a conversation turn.

    ── MODE A (Copilot Decides) ──────────────────────────────────────────────
    This endpoint is called autonomously by Copilot when it judges the user's
    question warrants a vector search. The result is returned to Copilot which
    then incorporates it into its final response.

    ── MODE B (Always Search) ────────────────────────────────────────────────
    JOE's sessionController can call this endpoint DIRECTLY via HTTP before
    sending the user's message to Copilot. See:
      copilot_bridge/src/controllers/sessionController.ts  →  sendMessageController()
    The returned context is prepended to the prompt as a system message block.
    """
    logger.info(f"Tool call: {request.name} | params: {request.parameters}")

    if request.name == "vector_search":
        return await _tool_vector_search(request.parameters)

    elif request.name == "list_sources":
        return await _tool_list_sources()

    else:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {request.name}")


# ─── Tool Implementations ─────────────────────────────────────────────────────

async def _tool_vector_search(params: dict) -> dict:
    """
    Core semantic search tool.
    Embeds the query using the BGE model, runs cosine similarity search
    against pgvector, and returns enriched results with neighboring chunks.
    """
    query = params.get("query", "").strip()
    if not query:
        return {"error": "query parameter is required"}

    top_k = min(int(params.get("top_k", 5)), 10)  # cap at 10 for safety
    include_context = params.get("include_context", True)

    if embedder is None:
        return {"error": "Embedding model not ready yet. Try again in a moment."}

    try:
        # Step 1: Embed the query using BGE query prefix
        # embed_query() applies "Represent this sentence for searching relevant passages: "
        # prefix automatically — this is the BGE-specific trick for better retrieval.
        query_vector = embedder.embed_query(query)

        # Step 2: Search pgvector
        with DatabaseManager() as db:
            if include_context:
                # Returns top chunks + their neighboring chunks from the same document.
                # context_window=1 means 1 chunk before + 1 chunk after each match.
                results = db.search_with_context(query_vector, top_k=top_k, context_window=1)
            else:
                # Simple vector similarity search with no neighbor expansion.
                results = db.search_similar(query_vector, top_k=top_k)

        if not results:
            return {
                "content": "No relevant documents found in the vector database for this query.",
                "results": []
            }

        # Step 3: Format results into a readable text block for Copilot to consume.
        # The text format matters — Copilot will read this as tool output and synthesize
        # a response. Clear structure = better synthesis.
        formatted_lines = [
            f"Vector search results for: \"{query}\"",
            f"Found {len(results)} relevant chunk(s).\n",
        ]

        for i, r in enumerate(results, 1):
            similarity_pct = round(r["similarity"] * 100, 1)

            # Build the Jabil document URL using doc_name and plant_code from metadata
            # Pattern: https://jdoc.jabil.com/app/SearchDocument/{doc_name}/{plant_code}
            doc_name = r.get("doc_name", "unknown")
            plant_code = r.get("plant_code", "unknown")
            if doc_name != "unknown" and plant_code != "unknown":
                doc_url = f"https://jdoc.jabil.com/app/SearchDocument/{doc_name}/{plant_code}"
                source_label = f"[{doc_name}]({doc_url})"
            else:
                source_label = r['source_file']

            formatted_lines.append(f"--- Result {i} | Source: {source_label} | Similarity: {similarity_pct}% ---")

            # Include neighboring context if available
            for ctx in r.get("context_before", []):
                formatted_lines.append(f"[Context before]\n{ctx}")

            formatted_lines.append(f"[Matched]\n{r['content']}")

            for ctx in r.get("context_after", []):
                formatted_lines.append(f"[Context after]\n{ctx}")

            formatted_lines.append("")  # blank line between results

        content_text = "\n".join(formatted_lines)

        return {
            # 'content' is the text Copilot reads as the tool result
            "content": content_text,
            # 'results' is the raw structured data — useful for MODE B (direct injection)
            "results": [
                {
                    "source_file": r["source_file"],
                    "doc_name": r.get("doc_name", "unknown"),
                    "plant_code": r.get("plant_code", "unknown"),
                    "doc_url": (
                        f"https://jdoc.jabil.com/app/SearchDocument/{r['doc_name']}/{r['plant_code']}"
                        if r.get("doc_name", "unknown") != "unknown"
                        else None
                    ),
                    "similarity": round(r["similarity"], 4),
                    "content": r["content"],
                    "full_context": r.get("full_context", r["content"]),
                    "chunk_type": r.get("chunk_type", "text"),
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"vector_search failed: {e}", exc_info=True)
        return {"error": f"Search failed: {str(e)}"}


async def _tool_list_sources() -> dict:
    """
    Lists all ingested documents from the vector DB.
    Returns a summary table of file names, types, chunk counts, and status.
    """
    try:
        with DatabaseManager() as db:
            summary = db.get_ingestion_summary()

        if not summary:
            return {"content": "No documents have been ingested into the vector database yet."}

        lines = ["Ingested documents in vector database:\n"]
        for doc in summary:
            lines.append(
                f"  • {doc['file_name']}  [{doc['file_type']}]  "
                f"{doc['total_chunks'] or 0} chunks  —  {doc['status']}"
            )

        return {
            "content": "\n".join(lines),
            "sources": [
                {
                    "id": doc["id"] if "id" in doc else None,
                    "file_name": doc["file_name"],
                    "file_type": doc["file_type"],
                    "total_chunks": doc["total_chunks"] or 0,
                    "status": doc["status"],
                }
                for doc in summary
            ]
        }

    except Exception as e:
        logger.error(f"list_sources failed: {e}", exc_info=True)
        return {"error": f"Failed to list sources: {str(e)}"}


# ─── Direct Search Endpoint (used by MODE B — Always Search) ─────────────────
# JOE's sessionController calls this endpoint directly to get RAG context
# BEFORE sending the user message to Copilot. This bypasses MCP tool calling
# entirely — the context is injected manually into the Copilot session prompt.
# See sessionController.ts → sendMessageController() for how this is consumed.

class DirectSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    include_context: bool = True


@app.post("/search")
async def direct_search(request: DirectSearchRequest):
    """
    Direct search endpoint for MODE B (Always Search / forced RAG).

    Called by copilot_bridge before sending user message to Copilot.
    Returns results in the same format as the MCP tool call above.

    Example usage in sessionController.ts:
        const ragContext = await fetch("http://localhost:8000/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: prompt, top_k: 5 })
        });
        const { results } = await ragContext.json();
        // inject results into prompt before session.send()
    """
    result = await _tool_vector_search({
        "query": request.query,
        "top_k": request.top_k,
        "include_context": request.include_context,
    })
    return result


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick health check. Returns model name and DB chunk count."""
    try:
        with DatabaseManager() as db:
            stats = db.get_stats()

        return {
            "status": "ok",
            "embedding_model": embedder.get_model_name() if embedder else "not loaded",
            "embedding_dimensions": embedder.get_dimensions() if embedder else None,
            "total_sources": stats["total_sources"],
            "total_chunks": stats["total_chunks"],
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
