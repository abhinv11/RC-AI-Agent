from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field


# 1) ROOT STATE
class State(TypedDict):
    # INPUT
    user_query: str
    parsed_inputs: Dict[str, Any]

    # ROUTER
    intent: str
    route: Literal["simple", "rag", "complex"]
    needs_research: bool

    # ORCHESTRATOR
    plan: Optional["ExecutionPlan"]

    # WORKERS
    design: Optional["DesignResult"]
    rebar: Optional["RebarResult"]
    cost: Optional["CostResult"]
    code: Optional["CodeCheckResult"]
    research: Optional["ResearchResult"]

    # REDUCER
    reduced: Optional["ReducedOutput"]

    # CRITIC
    critic: Optional["CriticResult"]
    replan_done: bool

    # IMAGE PIPELINE
    md_merged: str
    md_with_images: str
    image_specs: List[dict]

    # FINAL
    final_output: Optional[str]


# 2) ROUTER
class RouterDecision(BaseModel):
    intent: Literal["chat", "rc_design", "cost_query", "code_check"]
    route: Literal["simple", "rag", "complex"]
    needs_research: bool
    reason: str


# 3) ORCHESTRATOR
class PlanInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: Optional[str] = None
    span: Optional[float] = None
    load: Optional[float] = None
    moment: Optional[float] = None
    fck: Optional[float] = None
    fy: Optional[float] = None


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_type: Literal["beam", "slab", "column", "footing"]
    inputs: PlanInputs
    optimization_goal: Literal["min_cost", "min_steel", "balanced"]
    workers: List[Literal["design", "rebar", "cost", "code", "research"]]
    parallel_groups: List[List[str]]
    notes: List[str] = Field(default_factory=list)


# 4) WORKERS
class DesignResult(BaseModel):
    b: float
    d: float
    As: float
    moment_required: float
    moment_capacity: float
    utilization: float
    method: str


class RebarResult(BaseModel):
    bar_dia: float
    num_bars: int
    arrangement: str
    provided_As: float
    spacing: Optional[float]
    remarks: List[str]


class CostResult(BaseModel):
    concrete_volume: float
    steel_weight: float
    cost_concrete: float
    cost_steel: float
    total_cost: float
    optimality_note: str


class CodeCheckResult(BaseModel):
    flexure_ok: bool
    min_steel_ok: bool
    overall_safe: bool
    utilization: float
    warnings: List[str]


class ResearchResult(BaseModel):
    insights: List[str]
    sources: List[str]


# 5) REDUCER
class ReducedOutput(BaseModel):
    dimensions: Dict[str, float]
    steel: Dict[str, Any]
    cost: float
    safety: Literal["SAFE", "UNSAFE", "MARGINAL"]
    conflicts: List[str]
    combined_summary: str


# 6) CRITIC
class CriticResult(BaseModel):
    is_valid: bool
    issues: List[str]
    fix_suggestions: List[str]
    should_replan: bool


# 7) IMAGE PIPELINE
class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"


class ImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec]


# 8) FINAL REPORT
class FinalReport(BaseModel):
    title: str
    summary: str
    design: Dict[str, float]
    reinforcement: Dict[str, Any]
    cost_table: Dict[str, float]
    safety_status: str
    notes: List[str]
    images: Optional[List[str]]
