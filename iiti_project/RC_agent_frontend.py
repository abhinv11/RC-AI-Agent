from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import streamlit as st

from RC_agent import app
from rc_tools import ingest_markdown, thread_document_metadata


# -----------------------------
# Helpers
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            current_state.update(next(iter(step_payload.values())))
        else:
            current_state.update(step_payload)
    return current_state


def as_dict(maybe_obj: Any) -> Dict[str, Any]:
    if maybe_obj is None:
        return {}
    if isinstance(maybe_obj, dict):
        return maybe_obj
    if hasattr(maybe_obj, "model_dump"):
        return maybe_obj.model_dump()
    try:
        return json.loads(json.dumps(maybe_obj, default=str))
    except Exception:
        return {"value": str(maybe_obj)}


def _resolve_image_path(src: str) -> Path:
    return Path(src.strip().lstrip("./")).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            if payload.strip():
                st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: {src} (looked for {img_path})")

        i += 1


def generate_thread_id() -> str:
    return f"{st.session_state['session_namespace']}-{uuid.uuid4().hex}"


def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history_by_thread"].setdefault(thread_id, [])
    st.session_state["thread_outputs"].setdefault(thread_id, None)
    st.session_state["thread_logs"].setdefault(thread_id, [])


def history_for_thread(thread_id: str) -> List[Dict[str, str]]:
    return st.session_state["message_history_by_thread"].setdefault(thread_id, [])


def build_memory_prompt(thread_history: List[Dict[str, str]], user_input: str, max_turns: int = 6) -> str:
    recent = thread_history[-max_turns:]
    lines: List[str] = []
    for msg in recent:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        lines.append(f"{role.upper()}: {content}")

    convo = "\n".join(lines)
    if convo.strip():
        return (
            "Use prior chat context if relevant.\n\n"
            f"Conversation memory:\n{convo}\n\n"
            f"Current user query:\n{user_input}"
        )

    return user_input


def build_inputs(query_text: str, thread_id: str, needs_research_hint: bool, span: float, load: float, moment: float, fck: float, fy: float) -> Dict[str, Any]:
    parsed_inputs: Dict[str, Any] = {
        "thread_id": thread_id,
        "span": float(span),
        "load": float(load),
        "fck": float(fck),
        "fy": float(fy),
    }
    if moment > 0:
        parsed_inputs["moment"] = float(moment)

    return {
        "user_query": query_text,
        "parsed_inputs": parsed_inputs,
        "intent": "",
        "route": "simple",
        "needs_research": bool(needs_research_hint),
        "plan": None,
        "design": None,
        "rebar": None,
        "cost": None,
        "code": None,
        "research": None,
        "reduced": None,
        "critic": None,
        "replan_done": False,
        "md_merged": "",
        "md_with_images": "",
        "image_specs": [],
        "final_output": None,
    }


# -----------------------------
# Session init
# -----------------------------
st.set_page_config(page_title="RC Agent + Markdown RAG", layout="wide")
st.title("RC Agent + Markdown RAG")
st.caption("Thread-based RC chat with per-thread memory and markdown retrieval")

if "session_namespace" not in st.session_state:
    st.session_state["session_namespace"] = uuid.uuid4().hex
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"{st.session_state['session_namespace']}-{uuid.uuid4().hex}"
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []
if "message_history_by_thread" not in st.session_state:
    st.session_state["message_history_by_thread"] = {}
if "thread_outputs" not in st.session_state:
    st.session_state["thread_outputs"] = {}
if "thread_logs" not in st.session_state:
    st.session_state["thread_logs"] = {}
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_id = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_id, {})
thread_history = history_for_thread(thread_id)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Session")
    st.markdown(f"Thread ID: {thread_id}")

    if st.button("New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.subheader("Upload Markdown")
    uploaded_md = st.file_uploader("Upload markdown for this thread", type=["md", "markdown", "txt"])
    if uploaded_md:
        if uploaded_md.name in thread_docs:
            st.info(f"{uploaded_md.name} already indexed for this thread.")
        else:
            with st.status("Indexing markdown...", expanded=True) as status_box:
                try:
                    summary = ingest_markdown(
                        uploaded_md.getvalue(),
                        thread_id=thread_id,
                        filename=uploaded_md.name,
                    )
                    thread_docs[uploaded_md.name] = summary
                    status_box.update(label="Markdown indexed", state="complete", expanded=False)
                except Exception as e:
                    status_box.update(label="Index failed", state="error", expanded=True)
                    st.error(str(e))

    doc_meta = thread_document_metadata(thread_id)
    if doc_meta:
        st.success(
            f"Using {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, docs: {doc_meta.get('documents')})"
        )
    else:
        st.info("No markdown indexed yet for this thread.")

    st.subheader("RC Defaults")
    span = st.number_input("Span (m)", min_value=0.0, value=5.0, step=0.1)
    load = st.number_input("Load (kN/m)", min_value=0.0, value=20.0, step=0.5)
    moment = st.number_input("Moment (kN-m, optional)", min_value=0.0, value=0.0, step=1.0)
    fck = st.number_input("fck (MPa)", min_value=10.0, value=25.0, step=1.0)
    fy = st.number_input("fy (MPa)", min_value=250.0, value=500.0, step=10.0)
    needs_research_hint = st.checkbox("Hint: needs research", value=False)

    st.subheader("Past Conversations")
    for tid in st.session_state["chat_threads"][::-1]:
        if st.button(tid, key=f"thread-{tid}"):
            st.session_state["thread_id"] = tid
            st.rerun()


# -----------------------------
# Main chat
# -----------------------------
for message in thread_history:
    with st.chat_message(message["role"]):
        text = message.get("content", "")
        if isinstance(text, str) and text.lstrip().startswith("#"):
            render_markdown_with_local_images(text)
        else:
            st.write(text)

user_input = st.chat_input("Ask RC question, design query, or ask from uploaded markdown")

if user_input:
    thread_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    query_text = build_memory_prompt(thread_history[:-1], user_input)
    inputs = build_inputs(
        query_text=query_text,
        thread_id=thread_id,
        needs_research_hint=needs_research_hint,
        span=span,
        load=load,
        moment=moment,
        fck=fck,
        fy=fy,
    )

    logs: List[str] = []
    final_out: Dict[str, Any] = {}

    with st.chat_message("assistant"):
        status = st.status("Running RC graph...", expanded=True)
        progress = st.empty()

        current_state: Dict[str, Any] = {}
        last_node: Optional[str] = None

        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))

                if node_name and node_name != last_node:
                    status.write(f"Node: {node_name}")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)
                progress.json(
                    {
                        "intent": current_state.get("intent"),
                        "route": current_state.get("route"),
                        "has_plan": current_state.get("plan") is not None,
                        "has_reduced": current_state.get("reduced") is not None,
                        "images": len(current_state.get("image_specs") or []),
                    }
                )

                logs.append(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                final_out = as_dict(payload)
                status.update(label="Done", state="complete", expanded=False)
                logs.append("[final] received final state")

        assistant_text = str(final_out.get("final_output") or "No final output returned.")
        if assistant_text.lstrip().startswith("#"):
            render_markdown_with_local_images(assistant_text)
        else:
            st.write(assistant_text)

    thread_history.append({"role": "assistant", "content": assistant_text})
    st.session_state["thread_outputs"][thread_id] = final_out
    st.session_state["thread_logs"].setdefault(thread_id, []).extend(logs)


# -----------------------------
# Details tabs
# -----------------------------
out = st.session_state["thread_outputs"].get(thread_id)

summary_tab, pipeline_tab, images_tab, logs_tab = st.tabs(["Summary", "Pipeline", "Images", "Logs"])

with summary_tab:
    st.subheader("Latest Run Summary")
    if not out:
        st.info("No run yet in this thread.")
    else:
        cols = st.columns(4)
        cols[0].metric("Intent", str(out.get("intent", "-")))
        cols[1].metric("Route", str(out.get("route", "-")))
        cols[2].metric("Safety", str(as_dict(out.get("reduced")).get("safety", "-")))
        cols[3].metric("Images", str(len(out.get("image_specs") or [])))

        st.write("Parsed inputs")
        st.json(as_dict(out.get("parsed_inputs")))

with pipeline_tab:
    st.subheader("Latest Pipeline Output")
    if not out:
        st.info("No run yet in this thread.")
    else:
        st.write("Plan")
        st.json(as_dict(out.get("plan")))

        c1, c2 = st.columns(2)
        with c1:
            st.write("Design")
            st.json(as_dict(out.get("design")))
            st.write("Rebar")
            st.json(as_dict(out.get("rebar")))
        with c2:
            st.write("Cost")
            st.json(as_dict(out.get("cost")))
            st.write("Code Check")
            st.json(as_dict(out.get("code")))

        st.write("Research")
        st.json(as_dict(out.get("research")))

        st.write("Reduced")
        st.json(as_dict(out.get("reduced")))

with images_tab:
    st.subheader("Generated Images")
    if not out:
        st.info("No run yet in this thread.")
    else:
        if out.get("image_specs"):
            st.write("Image plan")
            st.json(out.get("image_specs"))

        images_dir = Path("images")
        if images_dir.exists() and images_dir.is_dir():
            files = sorted([p for p in images_dir.iterdir() if p.is_file()])
            if files:
                for p in files:
                    st.image(str(p), caption=p.name, use_container_width=True)
            else:
                st.info("images/ exists but no files were generated.")
        else:
            st.info("No local images directory found.")

with logs_tab:
    st.subheader("Thread Logs")
    st.text_area(
        "Events",
        value="\n\n".join(st.session_state["thread_logs"].get(thread_id, [])[-120:]),
        height=420,
    )
