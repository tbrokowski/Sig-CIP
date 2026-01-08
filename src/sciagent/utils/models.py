"""
Core data models for SciAgent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import asyncio


class ProcessState(Enum):
    """State of an experiment process"""
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HumanApprovalRequest:
    """Request sent to user for approval/decision"""
    request_id: str
    request_type: str  # "approval", "choice", "clarification", "review"
    context: Dict[str, Any]
    options: Optional[List[str]] = None
    timeout: Optional[int] = None
    default: Optional[str] = None
    response: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class Hypothesis:
    """Scientific hypothesis"""
    statement: str
    rationale: str
    testable_predictions: List[str]
    expected_evidence: str
    confidence: float = 0.0
    prior_probability: float = 0.5


@dataclass
class Paper:
    """Scientific paper metadata"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    url: Optional[str] = None
    year: Optional[int] = None
    citations: int = 0
    key_findings: List[str] = field(default_factory=list)


@dataclass
class DatasetRequirements:
    """Requirements for dataset preparation"""
    name: str
    version: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    transforms: Optional[List[str]] = None
    cache_dir: Optional[Path] = None


@dataclass
class ExperimentDesign:
    """Detailed experiment design"""
    description: str
    measurements: List[str]
    controls: List[str]
    sample_size: int
    statistical_tests: List[str]
    potential_confounds: List[str]
    success_criteria: str


@dataclass
class ExperimentSequence:
    """Sequence of experiments planned by MCTS"""
    steps: List[ExperimentDesign]
    total_value: float
    expected_cost: float
    expected_duration: float


@dataclass
class ExperimentPlan:
    """Complete experiment plan"""
    hypothesis: Hypothesis
    design: ExperimentDesign
    sequence: ExperimentSequence
    expected_value: float
    requires_dataset: bool = False
    dataset_requirements: Optional[DatasetRequirements] = None


@dataclass
class ExperimentPlanning:
    """Complete planning output from Science Agent"""
    query: str
    papers: List[Paper]
    hypotheses: List[Hypothesis]
    plans: List[ExperimentPlan]
    recommendation: ExperimentPlan


@dataclass
class CodeArtifact:
    """Generated code with metadata"""
    code: str
    quality_score: float
    iterations: int = 1
    failed: bool = False
    reflection: Optional[str] = None
    parent: Optional[CodeArtifact] = None
    variation_type: Optional[str] = None
    tests: Optional[str] = None


@dataclass
class TestResult:
    """Result of code testing"""
    success: bool
    error: Optional[str] = None
    output: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Critique:
    """Self-critique of code"""
    quality_score: float
    dimension_scores: Dict[str, float]
    feedback: str


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    data: Any
    summary: str
    metrics: Dict[str, float] = field(default_factory=dict)
    plots: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    traceback: Optional[str] = None
    duration: float = 0.0


@dataclass
class DataPreparation:
    """Result of data preparation"""
    dataset_name: str
    cache_path: Path
    loaders: Dict[str, Any]
    loader_code: str
    statistics: Dict[str, Any]
    samples: List[Any] = field(default_factory=list)


@dataclass
class ScientificAnalysis:
    """Analysis of experimental results"""
    summary: str
    supports_hypothesis: bool
    statistical_significance: float
    effect_size: float
    limitations: List[str]
    implications: List[str]
    confidence: float


@dataclass
class Refinement:
    """Proposed experiment refinement"""
    description: str
    rationale: str
    expected_information_gain: float
    cost: float


@dataclass
class ConstitutionalReview:
    """Review against scientific principles"""
    approved: bool
    violations: List[Dict[str, Any]]
    critical_violations: List[Dict[str, Any]]


@dataclass
class StatisticalReview:
    """Statistical validity review"""
    approved: bool
    issues: List[str]
    recommendations: List[str]


@dataclass
class DebateResult:
    """Result of multi-agent debate"""
    approved: bool
    confidence: float
    synthesis: str
    debate_history: List[Dict[str, Any]]
    points_of_agreement: List[str]
    points_of_contention: List[str]


@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    approved: bool
    confidence: float
    debate_summary: str
    constitutional_violations: List[Dict[str, Any]]
    statistical_issues: List[str]
    recommendations: List[str]


@dataclass
class Checkpoint:
    """Experiment checkpoint for pause/resume"""
    name: str
    timestamp: datetime
    data: Any


@dataclass
class ExperimentState:
    """State of a running experiment"""
    experiment_id: str
    query: str
    status: ProcessState
    checkpoints: List[Checkpoint] = field(default_factory=list)
    error: Optional[str] = None
    refinement_experiments: List[asyncio.Task] = field(default_factory=list)

    def add_checkpoint(self, name: str, data: Any) -> None:
        """Add a checkpoint"""
        self.checkpoints.append(Checkpoint(
            name=name,
            timestamp=datetime.now(),
            data=data
        ))


@dataclass
class ExperimentResult:
    """Final experiment result"""
    experiment_id: str
    query: str
    planning: ExperimentPlanning
    code: CodeArtifact
    execution: ExecutionResult
    analysis: ScientificAnalysis
    validation: ValidationResult
    refinements: List[Refinement]
    state: ExperimentState


@dataclass
class ProcessInfo:
    """Information about a background process"""
    process_id: str
    experiment_id: str
    status: ProcessState
    started_at: datetime
    completed_at: Optional[datetime] = None


# Constitutional AI principles for scientific research
SCIENTIFIC_CONSTITUTION = {
    "reproducibility": {
        "description": "Results must be reproducible with provided code and data",
        "severity": "critical"
    },
    "transparency": {
        "description": "Methods, assumptions, and limitations must be clearly stated",
        "severity": "critical"
    },
    "statistical_rigor": {
        "description": "Appropriate statistical tests with correct assumptions",
        "severity": "critical"
    },
    "ethical_data": {
        "description": "Data must be ethically sourced and properly attributed",
        "severity": "critical"
    },
    "bias_awareness": {
        "description": "Potential biases in data or methods must be acknowledged",
        "severity": "high"
    },
    "negative_results": {
        "description": "Negative results must be reported honestly",
        "severity": "high"
    },
    "peer_review": {
        "description": "Claims should be validated by multiple methods",
        "severity": "medium"
    },
    "version_control": {
        "description": "Code and experiments should be version controlled",
        "severity": "medium"
    }
}
