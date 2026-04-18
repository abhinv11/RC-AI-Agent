from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


_thread_retrievers: Dict[str, Any] = {}
_thread_metadata: Dict[str, Dict[str, Any]] = {}
_last_image_error: str = ""


def get_last_image_generation_error() -> str:
    return _last_image_error


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _thread_retrievers:
        return _thread_retrievers[thread_id]
    return None


def ingest_markdown(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a thread-specific retriever for uploaded markdown text.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    text = file_bytes.decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError("Uploaded markdown file is empty.")

    docs = [
        Document(
            page_content=text,
            metadata={"source": filename or "uploaded.md", "type": "markdown"},
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    try:
        from langchain_community.vectorstores import FAISS
    except Exception as exc:
        raise RuntimeError(
            "FAISS backend is unavailable. Install faiss-cpu to enable markdown RAG ingestion."
        ) from exc

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    key = str(thread_id)
    _thread_retrievers[key] = retriever
    _thread_metadata[key] = {
        "filename": filename or "uploaded.md",
        "documents": len(docs),
        "chunks": len(chunks),
    }

    return {
        "filename": filename or "uploaded.md",
        "documents": len(docs),
        "chunks": len(chunks),
    }


def thread_document_metadata(thread_id: str) -> dict:
    return _thread_metadata.get(str(thread_id), {})


def markdown_rag_search(query: str, thread_id: Optional[str] = None) -> dict:
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No markdown document indexed for this chat. Upload a markdown file first.",
            "query": query,
            "context": [],
            "metadata": [],
            "source_file": None,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _thread_metadata.get(str(thread_id), {}).get("filename"),
    }


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from uploaded markdown for this chat thread.
    Always include thread_id when calling this tool.
    """
    return markdown_rag_search(query=query, thread_id=thread_id)


tools = [search_tool, rag_tool]
llm_with_tools = llm.bind_tools(tools)


def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content") or r.get("snippet") or "",
            }
            for r in (results or [])
            if r.get("url")
        ]
    except Exception:
        return []


def optimize_rc_demo(
    M: float,
    fck: float = 25.0,
    fy: float = 500.0,
    target_utilization: float = 0.85,
    span: float = 1.0,
    objective: str = "min_cost",
) -> dict:
    """
    Simple RC beam demo optimizer.
    Returns section (b, d), steel area As, and basic utilization metrics.
    """
    M = max(float(M), 1.0)  # kN-m (factored design moment)
    fck = max(float(fck), 15.0)
    fy = max(float(fy), 250.0)
    target_utilization = min(max(float(target_utilization), 0.6), 0.95)
    span = max(float(span), 1.0)

    # Add design reserve so optimized sections are not unrealistically at 100% utilization.
    M_target = M / target_utilization

    best: Optional[Dict[str, float]] = None
    for b_mm in [250.0, 300.0, 350.0, 400.0]:
        for d_mm in range(400, 801, 25):
            d_mm = float(d_mm)
            Mu_nmm = M_target * 1e6

            z_mm = 0.9 * d_mm
            As_req_mm2 = Mu_nmm / max(0.87 * fy * z_mm, 1e-9)
            As_min_mm2 = (0.85 * b_mm * d_mm) / max(fy, 1e-9)
            As_mm2 = max(As_req_mm2, As_min_mm2)

            xu = (0.87 * fy * As_mm2) / max(0.36 * fck * b_mm, 1e-9)
            xu_lim = 0.48 * d_mm
            xu_eff = min(xu, xu_lim)
            z_eff_mm = d_mm - 0.42 * xu_eff
            Mu_cap_kNm = (0.87 * fy * As_mm2 * z_eff_mm) / 1e6

            # Must satisfy target-capacity requirement.
            if Mu_cap_kNm < M_target:
                continue

            b_m = b_mm / 1000.0
            d_m = d_mm / 1000.0
            As_m2 = As_mm2 / 1e6

            concrete_volume = b_m * d_m * span
            steel_weight = As_m2 * span * 7850.0
            concrete_rates = {20: 5500, 25: 6200, 30: 7000, 35: 7800, 40: 8500}
            c_rate = concrete_rates.get(int(fck), 6200)
            total_cost = concrete_volume * c_rate + steel_weight * 65.0

            metric = total_cost
            if objective == "min_steel":
                metric = steel_weight
            elif objective == "balanced":
                metric = 0.7 * total_cost + 0.3 * steel_weight

            if best is None or metric < best["metric"]:
                best = {
                    "b": b_m,
                    "d": d_m,
                    "As": As_m2,
                    "moment_capacity": Mu_cap_kNm,
                    "total_cost": total_cost,
                    "metric": metric,
                }

    if best is None:
        best = {
            "b": 0.3,
            "d": 0.5,
            "As": 0.001,
            "moment_capacity": 1.0,
            "total_cost": 0.0,
            "metric": 0.0,
        }

    b = best["b"]
    d = best["d"]
    As = best["As"]
    moment_capacity = best["moment_capacity"]
    utilization = M / max(moment_capacity, 1e-6)

    return {
        "b": b,
        "d": d,
        "As": As,
        "moment_required": M,
        "moment_capacity": moment_capacity,
        "utilization": utilization,
        "method": f"grid_search_bd_{objective}",
    }


def gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Generate image using Gemini.
    Requires: pip install google-genai
    Env: GOOGLE_API_KEY or GEMINI_API_KEY
    """
    global _last_image_error
    _last_image_error = ""

    try:
        from google import genai
        from google.genai import types
    except Exception:
        _last_image_error = "Gemini SDK import failed"
        print("DEBUG: Gemini SDK import failed")
        return b""

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    print("DEBUG: Generating image...")
    print(f"DEBUG: Prompt: {prompt}")
    print(f"DEBUG: GEMINI/GOOGLE API KEY: {'FOUND' if api_key else 'MISSING'}")
    if not api_key:
        _last_image_error = "GOOGLE_API_KEY/GEMINI_API_KEY is not set"
        print("DEBUG: API key not found in environment. Set GOOGLE_API_KEY or GEMINI_API_KEY")
        return b""

    try:
        client = genai.Client(api_key=api_key)
        print("DEBUG: Gemini client created successfully")
    except Exception as e:
        _last_image_error = f"Failed to create Gemini client: {type(e).__name__}: {e}"
        print(f"DEBUG: Failed to create Gemini client: {e}")
        return b""

    resp = None
    image_models = [
        "models/gemini-2.5-flash-image",
        "models/gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image",
    ]
    for model_name in image_models:
        try:
            print(f"DEBUG: Calling Gemini API for image generation with model: {model_name}")
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
            print(f"DEBUG: Gemini API response received from {model_name}: {type(resp)}")
            break
        except Exception as e:
            _last_image_error = f"Gemini API call failed for {model_name}: {type(e).__name__}: {e}"
            print(f"DEBUG: Gemini API call failed for {model_name}: {type(e).__name__}: {e}")

    if resp is None:
        if not _last_image_error:
            _last_image_error = "No response from Gemini image models"
        return b""

    try:
        parts = getattr(resp, "parts", None)
        if not parts and getattr(resp, "candidates", None):
            try:
                parts = resp.candidates[0].content.parts
            except Exception as e:
                print(f"DEBUG: Failed to extract parts from candidates: {e}")
                parts = None

        if not parts:
            _last_image_error = "Gemini returned no parts"
            print("DEBUG: Gemini returned no parts")
            return b""

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                print(f"DEBUG: Gemini image bytes received ({len(inline.data)} bytes)")
                return inline.data

        print("DEBUG: Gemini returned parts but no inline image bytes")
        print(f"DEBUG: Part types: {[type(p).__name__ for p in parts]}")
        _last_image_error = "Gemini returned parts but no inline image bytes"
        return b""
    except Exception as e:
        _last_image_error = f"Error processing Gemini response: {type(e).__name__}: {e}"
        print(f"DEBUG: Error processing Gemini response: {type(e).__name__}: {e}")
        return b""
