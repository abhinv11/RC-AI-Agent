from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from rc_state import ExecutionPlan, ResearchResult, RouterDecision, State
from rc_tools import (
    gemini_generate_image_bytes,
    get_last_image_generation_error,
    llm,
    markdown_rag_search,
    optimize_rc_demo,
    tavily_search,
)


# ============================================================
# RC Design Agent
# Start -> Router(simple/rag/complex) -> Orchestrator -> Workers -> Reducer
# Reducer image flow: merge_results -> decide_images -> generate_and_place_images
# ============================================================


ROUTER_SYSTEM = """
You are the Intent Router for an RC (Reinforced Concrete) Design Agent.

Your job is to classify the user query and decide how to route it.

Classify into:

1. chat
   - casual questions, explanations, theory
   - example: "what is bending moment?"

2. rc_design
   - full structural design problems
   - example: "design a beam for span 5m and load 20kN/m"

3. cost_query
   - cost estimation or optimization only
   - example: "how to reduce cost of concrete beam?"

4. code_check
   - validation against IS 456 rules
   - example: "is this design safe?"

Routing:

- simple:
    direct answer, no pipeline
- rag:
    needs external knowledge (rare)
- complex:
    requires full pipeline:
    Orchestrator -> Workers -> Reducer -> Critic

Rules:

- rc_design -> ALWAYS complex
- cost_query -> simple OR complex (depending on depth)
- code_check -> complex if inputs given
- chat -> simple
- Use rag ONLY if clearly external/latest info is needed

Also extract whether research is needed (usually false).

Return structured output only.
"""


RESEARCH_SYSTEM = """
You are a technical research assistant for an RC Design Agent.

You may be given raw web search results.

Your job:
- Extract useful engineering insights
- Identify typical values, rules of thumb, or missing assumptions
- Ignore fluff, blogs, and irrelevant content

Focus on:
- RC design heuristics (span/depth, steel ratios, etc.)
- Cost benchmarks (if relevant)
- Code-related clarifications (IS 456 style)

Rules:
- Output concise insights (bullet points)
- Do NOT summarize articles
- Do NOT include unnecessary citations
- Include sources ONLY if highly relevant

Return structured output.
"""


ORCH_SYSTEM = """
You are the Orchestrator for an RC (Reinforced Concrete) Design Agent.

Your job:
- Understand the engineering problem
- Extract structured inputs (span, loads, materials, etc.)
- Decide which workers are needed
- Create an execution plan

Workers available:
- design      -> section sizing (b, d, As)
- rebar       -> reinforcement layout
- cost        -> cost estimation/optimization
- code        -> safety checks (IS 456 style)
- research    -> only if inputs missing or unclear

Rules:

1. ALWAYS include "design" for rc_design problems
2. Include "cost" if optimization or cost is mentioned
3. Include "code" for safety validation
4. Include "rebar" after design
5. Include "research" ONLY if inputs are missing

Execution:
- design, cost, code can run in parallel
- rebar runs AFTER design

Optimization goals:
- min_cost -> prioritize cost
- balanced -> safe + reasonable cost

Keep plan minimal and practical.
Return structured ExecutionPlan only.
"""


def _extract_query_overrides(user_query: str) -> Dict[str, float]:
    full_q = user_query or ""
    q = full_q
    marker = "Current user query:"
    if marker in q:
        q = q.split(marker)[-1].strip()
    overrides: Dict[str, float] = {}

    span_hits = re.findall(r"span\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?)\s*m\b", q, flags=re.IGNORECASE)
    if span_hits:
        overrides["span"] = float(span_hits[-1])

    load_hits = re.findall(
        r"(?:uniform\s+load|udl|load)\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?)\s*kN\s*/\s*m",
        q,
        flags=re.IGNORECASE,
    )
    if load_hits:
        overrides["load"] = float(load_hits[-1])

    moment_hits = re.findall(r"moment\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?)\s*kN\s*-?m", q, flags=re.IGNORECASE)
    if moment_hits:
        overrides["moment"] = float(moment_hits[-1])

    fck_hits = re.findall(r"\bM\s*(\d+(?:\.\d+)?)\b", q, flags=re.IGNORECASE)
    if fck_hits:
        overrides["fck"] = float(fck_hits[-1])

    fy_hits = re.findall(r"\bFe\s*(\d+(?:\.\d+)?)\b", q, flags=re.IGNORECASE)
    if fy_hits:
        overrides["fy"] = float(fy_hits[-1])

    # If query is relative (e.g., "same problem as above"), backfill from latest USER memory lines.
    if marker in full_q and (
        "same problem" in q.lower()
        or "same beam" in q.lower()
        or "same design" in q.lower()
        or "as above" in q.lower()
        or "previous" in q.lower()
    ):
        memory_text = full_q.split(marker)[0]
        user_lines = re.findall(r"^USER:\s*(.+)$", memory_text, flags=re.IGNORECASE | re.MULTILINE)
        for line in reversed(user_lines):
            hist = _extract_query_overrides(line)
            for key in ["span", "load", "moment", "fck", "fy"]:
                if key not in overrides and key in hist:
                    overrides[key] = hist[key]
            if all(k in overrides for k in ["span", "load", "fck", "fy"]):
                break

    return overrides


def _extract_goal_override(user_query: str) -> str | None:
    q = (user_query or "")
    marker = "Current user query:"
    if marker in q:
        q = q.split(marker)[-1].strip()
    q = q.lower()
    if "minimum steel" in q or "min steel" in q:
        return "min_steel"
    if "minimum cost" in q or "min cost" in q or "optimize for cost" in q:
        return "min_cost"
    if "balanced" in q:
        return "balanced"
    return None


def _is_comparison_query(user_query: str) -> bool:
    q = (user_query or "")
    marker = "Current user query:"
    if marker in q:
        q = q.split(marker)[-1].strip()
    q = q.lower()
    return any(token in q for token in ["compare", "comparison", "delta", "difference", "previous result"])


def _extract_previous_metrics(user_query: str) -> Dict[str, float]:
    full_q = user_query or ""
    marker = "Current user query:"
    if marker not in full_q:
        return {}

    memory_text = full_q.split(marker)[0]
    assistant_lines = re.findall(r"^ASSISTANT:\s*(.+)$", memory_text, flags=re.IGNORECASE | re.MULTILINE)
    if not assistant_lines:
        return {}

    last_assistant = assistant_lines[-1]

    cost_hits = re.findall(r"Estimated\s+total\s+cost:\s*([0-9]+(?:\.[0-9]+)?)", last_assistant, flags=re.IGNORECASE)
    util_hits = re.findall(r"Utilization:\s*([0-9]+(?:\.[0-9]+)?)", last_assistant, flags=re.IGNORECASE)

    out: Dict[str, float] = {}
    if cost_hits:
        out["previous_cost"] = float(cost_hits[-1])
    if util_hits:
        out["previous_utilization"] = float(util_hits[-1])
    return out


def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)

    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"User Query: {state['user_query']}"),
        ]
    )

    return {
        "intent": decision.intent,
        "route": decision.route,
        "needs_research": decision.needs_research,
    }


def route_next(state: State) -> str:
    # Comparison prompts should use the full pipeline so reducer can compute deltas.
    if _is_comparison_query(state.get("user_query", "")):
        return "orchestrator"

    if state["route"] == "simple":
        return "direct_answer"

    if state["route"] == "rag":
        return "research"

    if state["route"] == "complex":
        return "orchestrator"

    return "direct_answer"


def direct_answer_node(state: State) -> dict:
    thread_id = (state.get("parsed_inputs") or {}).get("thread_id")
    rag = markdown_rag_search(query=state.get("user_query", ""), thread_id=thread_id)
    rag_context = "\n\n".join((rag.get("context") or [])[:3])

    user_prompt = state.get("user_query", "")
    if rag_context:
        user_prompt = (
            f"User Query: {state.get('user_query', '')}\n\n"
            f"Thread Markdown Context:\n{rag_context}\n\n"
            "Use this context when relevant."
        )

    reply = llm.invoke(
        [
            SystemMessage(content="You are a helpful RC engineering assistant. Answer briefly and clearly."),
            HumanMessage(content=user_prompt),
        ]
    )
    return {"final_output": reply.content}


def research_node(state: State) -> dict:
    if not state.get("needs_research"):
        return {"research": None}

    query = state.get("user_query", "")
    thread_id = (state.get("parsed_inputs") or {}).get("thread_id")

    rag = markdown_rag_search(query=query, thread_id=thread_id)
    local_context = rag.get("context") or []

    query_gen = llm.invoke(
        f"""
        Convert this into 3 focused engineering search queries:
        "{query}"

        Focus on:
        - RC design rules
        - cost optimization
        - IS 456 concepts (if relevant)

        Output as list.
        """
    )

    queries = query_gen.content.split("\n")[:3]

    raw_results = []
    for q in queries:
        raw_results.extend(tavily_search(q, max_results=3))

    researcher = llm.with_structured_output(ResearchResult)

    result = researcher.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=f"""
                User Query: {query}

                Inputs: {state.get('parsed_inputs')}

                Thread Markdown Context:
                {local_context}

                Web Results:
                {raw_results}
                """
            ),
        ]
    )

    return {"research": result}


def orchestrator_node(state: State) -> dict:
    user_query = state.get("user_query", "")
    parsed_inputs = dict(state.get("parsed_inputs") or {})

    # Keep parsed_inputs as single source of truth and override only when query explicitly provides values.
    parsed_inputs.update(_extract_query_overrides(user_query))

    planner = llm.with_structured_output(ExecutionPlan)

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=f"""
                User Query:
                {state['user_query']}

                Parsed Inputs:
                {parsed_inputs}

                Intent:
                {state.get('intent')}

                Research (if any):
                {state.get('research')}
                """
            ),
        ]
    )

    # Enforce consistent optimization goal when user explicitly asks for one.
    goal_override = _extract_goal_override(user_query)
    if goal_override:
        plan.optimization_goal = goal_override

    # For rc_design, ensure required worker coverage for stable report output.
    workers = list(plan.workers or [])
    for req in ["design", "rebar", "code"]:
        if req not in workers:
            workers.append(req)
    if plan.optimization_goal in ("min_cost", "balanced") and "cost" not in workers:
        workers.append("cost")
    plan.workers = workers

    if not plan.parallel_groups:
        parallel_base = [w for w in ["design", "cost", "code"] if w in workers]
        tail = [w for w in ["rebar", "research"] if w in workers]
        groups = []
        if parallel_base:
            groups.append(parallel_base)
        if tail:
            groups.append(tail)
        plan.parallel_groups = groups

    if plan and plan.inputs:
        for key in ["thread_id", "span", "load", "moment", "fck", "fy"]:
            if key in parsed_inputs:
                setattr(plan.inputs, key, parsed_inputs[key])

    return {"plan": plan, "parsed_inputs": parsed_inputs}


def worker_node(payload: dict) -> dict:
    worker_type = payload["worker_type"]

    inputs = payload.get("inputs", {})

    if worker_type == "design":
        span = float(inputs.get("span", 5.0) or 5.0)
        load = float(inputs.get("load", 10.0) or 10.0)
        # IS456 Cl.18.2: use 1.5 x (DL + LL) for factored design moment.
        # For simply supported beam under UDL, Mu = 1.5 * wL^2/8 (kN-m).
        M = float(inputs.get("moment") or (1.5 * load * span * span / 8.0))
        fck = float(inputs.get("fck", 25.0) or 25.0)
        fy = float(inputs.get("fy", 500.0) or 500.0)
        objective = str(payload.get("optimization_goal") or "min_cost")
        result = optimize_rc_demo(M=M, fck=fck, fy=fy, span=span, objective=objective)
        return {"design": result}

    if worker_type == "cost":
        b = payload.get("design", {}).get("b", 0.3)
        d = payload.get("design", {}).get("d", 0.5)
        As = payload.get("design", {}).get("As", 0.01)
        span = float(inputs.get("span", 1.0) or 1.0)

        concrete_volume = b * d * span
        steel_weight = As * span * 7850

        concrete_rates = {20: 5500, 25: 6200, 30: 7000, 35: 7800, 40: 8500}
        fck_val = float(inputs.get("fck", 25.0) or 25.0)
        c_rate = concrete_rates.get(int(fck_val), 6200)
        cost_concrete = concrete_volume * c_rate
        cost_steel = steel_weight * 65

        return {
            "cost": {
                "concrete_volume": concrete_volume,
                "steel_weight": steel_weight,
                "cost_concrete": cost_concrete,
                "cost_steel": cost_steel,
                "total_cost": cost_concrete + cost_steel,
                "optimality_note": "Demo estimate based on full beam quantities",
            }
        }

    if worker_type == "code":
        b = payload.get("design", {}).get("b", 0.3)
        d = payload.get("design", {}).get("d", 0.5)
        As = payload.get("design", {}).get("As", 0.01)
        fck = float(inputs.get("fck", 25.0) or 25.0)
        fy = float(inputs.get("fy", 500.0) or 500.0)

        span = float(inputs.get("span", 5.0) or 5.0)
        load = float(inputs.get("load", 10.0) or 10.0)
        M_required = float(inputs.get("moment") or (1.5 * load * span * span / 8.0))

        b_mm = b * 1000.0
        d_mm = d * 1000.0
        As_mm2 = As * 1e6

        xu = (0.87 * fy * As_mm2) / max(0.36 * fck * b_mm, 1e-9)
        xu_lim = 0.48 * d_mm
        xu_eff = min(xu, xu_lim)
        z = d_mm - 0.42 * xu_eff

        M_capacity = (0.87 * fy * As_mm2 * z) / 1e6

        safe = M_capacity >= M_required

        As_min_mm2 = (0.85 * b_mm * d_mm) / max(fy, 1e-9)
        min_steel_ok = As_mm2 >= As_min_mm2

        warnings = []
        if not safe:
            warnings.append("Insufficient moment capacity")
        if not min_steel_ok:
            warnings.append("Provided steel is below minimum steel requirement")

        return {
            "code": {
                "flexure_ok": safe,
                "min_steel_ok": min_steel_ok,
                "overall_safe": safe and min_steel_ok,
                "utilization": M_required / max(M_capacity, 1e-6),
                "warnings": warnings,
            }
        }

    if worker_type == "rebar":
        As_m2 = payload.get("design", {}).get("As", 0.001)
        As_mm2 = As_m2 * 1e6

        # IS standard bar diameters (IS 1786)
        is_bars = [8, 10, 12, 16, 20, 25, 32]

        best_dia, best_count, best_provided = 12, 4, 0.0
        for dia in is_bars:
            area_bar = math.pi * dia ** 2 / 4
            count = max(2, math.ceil(As_mm2 / area_bar))
            provided = count * area_bar
            if count <= 6:
                best_dia, best_count, best_provided = dia, count, provided
                break

        provided_m2 = best_provided * 1e-6

        return {
            "rebar": {
                "bar_dia": best_dia,
                "num_bars": best_count,
                "provided_As": provided_m2,
                "spacing": None,
                "arrangement": "bottom tension zone",
                "remarks": [
                    f"Required Ast = {As_mm2:.0f} mm2",
                    f"Provided: {best_count}-{best_dia}mm = {best_provided:.0f} mm2",
                    "IS 1786 bar sizes used",
                ],
            }
        }

    if worker_type == "research":
        return {
            "research": {
                "insights": [
                    "Typical span/depth ratio is 12-20",
                    "Minimum steel ratio ~0.2%",
                ],
                "sources": [],
            }
        }

    return {}


def workers_node(state: State) -> dict:
    plan = state.get("plan")
    if not plan:
        return {}

    out: Dict[str, Any] = {}
    if plan.inputs:
        inputs = plan.inputs.model_dump(exclude_none=True)
    else:
        inputs = {}
    workers = plan.workers or []

    design_payload = {
        "worker_type": "design",
        "inputs": inputs,
        "research": state.get("research"),
        "optimization_goal": getattr(plan, "optimization_goal", "min_cost"),
    }

    if "design" in workers:
        out.update(worker_node(design_payload))

    design_ctx = out.get("design") or state.get("design")

    ordered_rest = [w for w in ["cost", "code", "rebar", "research"] if w in workers]
    for w in ordered_rest:
        payload = {
            "worker_type": w,
            "inputs": inputs,
            "design": design_ctx,
            "research": state.get("research"),
            "optimization_goal": getattr(plan, "optimization_goal", "min_cost"),
        }
        out.update(worker_node(payload))

    return out


def merge_results(state: State) -> dict:
    design = state.get("design") or {}
    cost = state.get("cost") or {}
    code = state.get("code") or {}
    rebar = state.get("rebar") or {}
    plan = state.get("plan")
    parsed_inputs = state.get("parsed_inputs") or {}
    user_query = state.get("user_query", "")

    conflicts = []

    if code and not code.get("overall_safe", True):
        conflicts.append("Design is unsafe as per code check")

    if design and rebar and rebar.get("provided_As", 0) < design.get("As", 0):
        conflicts.append("Provided steel is less than required")

    if conflicts:
        safety = "UNSAFE"
    elif code.get("overall_safe", False):
        safety = "SAFE"
    else:
        safety = "MARGINAL"

    utilization = float(design.get("utilization", 0.0) or 0.0)
    moment_req = float(design.get("moment_required", 0.0) or 0.0)
    moment_cap = float(design.get("moment_capacity", 0.0) or 0.0)

    rebar_txt = (
        f"{rebar.get('num_bars', 0)} bars of {rebar.get('bar_dia', 0)} mm diameter "
        f"({rebar.get('arrangement', 'bottom tension')})"
    )
    provided_as_mm2 = float(rebar.get("provided_As", 0.0) or 0.0) * 1e6

    explanation = (
        f"Depth and width were selected to resist {moment_req:.2f} kN-m with "
        f"moment capacity {moment_cap:.2f} kN-m. "
        f"The chosen reinforcement gives utilization {utilization:.2f}."
    )

    objective = getattr(plan, "optimization_goal", "balanced") if plan else "balanced"
    if objective == "min_cost":
        explanation += " Objective was minimum cost under safety checks."
    elif objective == "min_steel":
        explanation += " Objective was minimum steel under safety checks."
    else:
        explanation += " Objective was a balanced cost-safety solution."

    used_inputs = {
        "span": parsed_inputs.get("span"),
        "load": parsed_inputs.get("load"),
        "moment": parsed_inputs.get("moment"),
        "fck": parsed_inputs.get("fck"),
        "fy": parsed_inputs.get("fy"),
    }

    comparison = None
    if _is_comparison_query(user_query):
        prev = _extract_previous_metrics(user_query)
        curr_cost = cost.get("total_cost")
        curr_util = utilization
        if prev and (curr_cost is not None):
            comparison = {
                "previous_cost": prev.get("previous_cost"),
                "current_cost": curr_cost,
                "delta_cost": (curr_cost - prev["previous_cost"]) if prev.get("previous_cost") is not None else None,
                "previous_utilization": prev.get("previous_utilization"),
                "current_utilization": curr_util,
                "delta_utilization": (
                    curr_util - prev["previous_utilization"]
                    if prev.get("previous_utilization") is not None
                    else None
                ),
            }

    return {
        "reduced": {
            "dimensions": {
                "b": design.get("b"),
                "d": design.get("d"),
            },
            "steel": rebar,
            "cost": cost.get("total_cost"),
            "safety": safety,
            "conflicts": conflicts,
            "combined_summary": "RC design completed with optimization and validation",
            "design_explanation": explanation,
            "reinforcement_summary": rebar_txt,
            "provided_As_mm2": provided_as_mm2,
            "design_utilization": utilization,
            "objective": objective,
            "used_inputs": used_inputs,
            "comparison": comparison,
        }
    }


def decide_images(state: State) -> dict:
    reduced = state["reduced"]

    images = []

    if reduced.get("dimensions"):
        images.append(
            {
                "placeholder": "[[IMAGE_1]]",
                "filename": "beam_section.png",
                "alt": "RC beam cross section",
                "caption": "Beam cross-section showing b, d and reinforcement",
                "prompt": (
                    "clean technical diagram of reinforced concrete beam cross section, "
                    "label width b, depth d, and bottom steel reinforcement bars, "
                    "minimal, engineering drawing style, white background"
                ),
            }
        )

    has_cost = reduced.get("cost") is not None
    if has_cost:
        images.append(
            {
                "placeholder": "[[IMAGE_2]]",
                "filename": "cost_tradeoff.png",
                "alt": "Cost vs depth tradeoff",
                "caption": "Trade-off between depth and cost",
                "prompt": (
                    "simple engineering graph showing cost vs depth curve, "
                    "U-shaped optimization curve, labeled axes, minimal clean style"
                ),
            }
        )

    cost_block = "- Estimated total cost: N/A (cost worker not requested)"
    if has_cost:
        cost_block = f"- Estimated total cost: {reduced.get('cost')}"

    image2_block = ""
    if has_cost:
        image2_block = "\n[[IMAGE_2]]\n"

    used = reduced.get("used_inputs") or {}
    comparison = reduced.get("comparison") or {}
    comparison_block = ""
    if comparison:
        comparison_block = (
            "\n## Comparison\n"
            f"- Previous cost: {comparison.get('previous_cost')}\n"
            f"- Current cost: {comparison.get('current_cost')}\n"
            f"- Delta cost: {comparison.get('delta_cost')}\n"
            f"- Previous utilization: {comparison.get('previous_utilization')}\n"
            f"- Current utilization: {comparison.get('current_utilization')}\n"
            f"- Delta utilization: {comparison.get('delta_utilization')}\n"
        )

    md = f"""
# RC Design Report

## Design Summary
{reduced.get('combined_summary')}

## Engineering Explanation
{reduced.get('design_explanation')}

## Dimensions
- Width (b): {reduced['dimensions'].get('b')}
- Depth (d): {reduced['dimensions'].get('d')}

[[IMAGE_1]]

## Reinforcement
- {reduced.get('reinforcement_summary')}
- Total steel area: {reduced.get('provided_As_mm2')} mm2

## Cost
{cost_block}
{image2_block}

## Safety
- Status: {reduced.get('safety')}
- Utilization: {reduced.get('design_utilization')}

## Assumptions
- Objective: {reduced.get('objective')}
- Used span (m): {used.get('span')}
- Used load (kN/m): {used.get('load')}
- Used moment (kN-m): {used.get('moment')}
- Used concrete grade fck (MPa): {used.get('fck')}
- Used steel grade fy (MPa): {used.get('fy')}

{comparison_block}

## Conflicts
{reduced.get('conflicts')}
"""

    return {
        "md_with_images": md,
        "image_specs": images[:2],
    }


def generate_and_place_images(state: State) -> dict:
    md = state.get("md_with_images", "")
    image_specs = state.get("image_specs", [])

    if not md:
        reduced = state.get("reduced") or {}
        md = (
            "# RC Design Report\n\n"
            f"Safety: {reduced.get('safety', 'MARGINAL')}\n\n"
            f"Summary: {reduced.get('combined_summary', 'No summary available.')}\n"
        )

    if not image_specs:
        return {"final_output": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    image_skip_reason = None
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        image_skip_reason = "Image generation skipped: GOOGLE_API_KEY/GEMINI_API_KEY is not set."

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        try:
            if not out_path.exists():
                img_bytes = gemini_generate_image_bytes(spec["prompt"])
                if img_bytes:
                    print(f"DEBUG: Saving image to: {out_path}")
                    out_path.write_bytes(img_bytes)
                else:
                    print(f"DEBUG: No image bytes returned for: {out_path}")

            if out_path.exists():
                img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            else:
                detailed_reason = get_last_image_generation_error().strip()
                reason = image_skip_reason or detailed_reason or "Image generation unavailable"
                img_md = f"*[{spec['caption']}] ({reason})*"
        except Exception:
            detailed_reason = get_last_image_generation_error().strip()
            reason = image_skip_reason or detailed_reason or "Image generation unavailable"
            img_md = f"*[{spec['caption']}] ({reason})*"

        md = md.replace(placeholder, img_md)

    return {"final_output": md}


reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_results", merge_results)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_results")
reducer_graph.add_edge("merge_results", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()


g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("direct_answer", direct_answer_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("workers", workers_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")

g.add_conditional_edges(
    "router",
    route_next,
    {
        "direct_answer": "direct_answer",
        "research": "research",
        "orchestrator": "orchestrator",
    },
)

g.add_edge("direct_answer", END)
g.add_edge("research", "orchestrator")
g.add_edge("orchestrator", "workers")
g.add_edge("workers", "reducer")
g.add_edge("reducer", END)

app = g.compile()
