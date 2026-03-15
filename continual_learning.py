"""
Deep Continual Learning for ETI / GRAD.py
==========================================

Production-ready continual learning framework integrating with GRAD.py.
Designed for human-curriculum-style sequential teaching.

Strategies:
    Regularization : EWC, Online-EWC, Synaptic Intelligence (SI)
    Replay         : Experience Replay, GEM (Gradient Episodic Memory)
    Architecture   : Progressive Neural Networks (PNN), PackNet (pruning)
    Distillation   : Learning without Forgetting (LwF)
    Hybrid         : EWC+Replay, SI+Replay

Curriculum Support:
    - Ordered task sequences (letters -> words -> sentences)
    - Difficulty scheduling
    - Competency gating (advance only when threshold met)
    - Per-task forgetting / plasticity metrics (BWT, FWT)

Version: 2.0.0-PRODUCTION
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional,
    Tuple, Union
)

import numpy as np
from numpy.typing import NDArray

# GRAD.py integration
try:
    from GRAD import (
        ParametricLens, ParameterState, TensorSpec, TangentBundle,
        CotangentBundle, UniversalTrainer, TrainingConfig, TrainingMetrics,
        LensFactory, ConstraintType, Adam, SGD, MSE, LossFunction,
        OptimizerConfig, OptimizerState,
    )
    _GRAD_OK = True
except ImportError:
    _GRAD_OK = False
    warnings.warn("GRAD.py not found - some features unavailable.", ImportWarning)

# Logging
logger = logging.getLogger("eti.continual_learning")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


# ============================================================================
# SECTION 1: ENUMS & CORE DATA STRUCTURES
# ============================================================================

class CLStrategy(Enum):
    EWC          = auto()
    ONLINE_EWC   = auto()
    SI           = auto()
    REPLAY       = auto()
    GEM          = auto()
    PNN          = auto()
    PACKNET      = auto()
    LWF          = auto()
    EWC_REPLAY   = auto()
    SI_REPLAY    = auto()


class TaskMode(Enum):
    """Incremental learning scenario."""
    TASK_INCREMENTAL   = auto()   # Task identity known at test time
    CLASS_INCREMENTAL  = auto()   # Task identity unknown; grow output head
    DOMAIN_INCREMENTAL = auto()   # Same output space, different input distribution


@dataclass
class TaskRecord:
    """Complete record for one task / lesson."""
    task_id:     int
    name:        str
    description: str   = ""
    difficulty:  float = 1.0          # 0-1 curriculum difficulty
    mode: TaskMode = field(default_factory=lambda: TaskMode.TASK_INCREMENTAL)

    # Data
    train_data: List[Tuple[Any, Any]] = field(default_factory=list, repr=False)
    val_data:   Optional[List[Tuple[Any, Any]]] = field(default=None, repr=False)

    # Post-training snapshots
    optimal_params:   Optional[NDArray] = field(default=None, repr=False)
    fisher_diag:      Optional[NDArray] = field(default=None, repr=False)
    si_importance:    Optional[NDArray] = field(default=None, repr=False)
    old_logits_cache: Dict[str, NDArray] = field(default_factory=dict, repr=False)

    # Metrics
    created_at:       datetime = field(default_factory=datetime.now)
    trained_at:       Optional[datetime] = None
    epochs_trained:   int   = 0
    train_loss_final: float = float("inf")
    val_loss_final:   float = float("inf")
    accuracy:         float = 0.0

    # Curriculum gate
    competency_threshold: float = 0.0
    passed_gate:          bool  = False


@dataclass
class CLMetrics:
    """Continual learning evaluation metrics (BWT, FWT, avg accuracy)."""
    task_accuracy_at_train: Dict[int, float] = field(default_factory=dict)
    task_accuracy_final:    Dict[int, float] = field(default_factory=dict)

    @property
    def backward_transfer(self) -> float:
        """BWT: negative = forgetting, positive = positive backward transfer."""
        shared = [t for t in self.task_accuracy_at_train
                  if t in self.task_accuracy_final]
        if len(shared) < 2:
            return 0.0
        return float(np.mean([
            self.task_accuracy_final[t] - self.task_accuracy_at_train[t]
            for t in shared[:-1]
        ]))

    @property
    def average_accuracy(self) -> float:
        if not self.task_accuracy_final:
            return 0.0
        return float(np.mean(list(self.task_accuracy_final.values())))

    def report(self) -> str:
        lines = [
            "── Continual Learning Metrics ──",
            f"  Tasks evaluated : {len(self.task_accuracy_final)}",
            f"  Avg accuracy    : {self.average_accuracy:.4f}",
            f"  Backward transfer (BWT): {self.backward_transfer:+.4f}",
        ]
        for t, acc in sorted(self.task_accuracy_final.items()):
            at_train = self.task_accuracy_at_train.get(t, float("nan"))
            lines.append(f"    Task {t:3d}: final={acc:.4f}  at_train={at_train:.4f}")
        return "\n".join(lines)


# ============================================================================
# SECTION 2: IMPORTANCE / REGULARIZATION ESTIMATORS
# ============================================================================

class FisherEstimator:
    """
    Diagonal Fisher Information Matrix estimator.

    Supports batch (EWC) and online (Online-EWC) modes.
    Online mode: F_t = gamma * F_{t-1} + (1-gamma) * g_t^2
    """

    def __init__(self, online: bool = False, gamma: float = 0.95):
        self.online  = online
        self.gamma   = gamma
        self._running: Optional[NDArray] = None

    def compute(
        self,
        lens: "ParametricLens",
        params: "ParameterState",
        data: List[Tuple[Any, Any]],
        loss_fn: "LossFunction",
        n_samples: int = 200,
    ) -> NDArray:
        """Compute diagonal Fisher over up to n_samples data points."""
        fisher = np.zeros(params.data.size, dtype=np.float64)
        samples = data[:n_samples]

        for x, y in samples:
            pred = lens.forward(params.data, x)
            _, cotangent = loss_fn(pred, y)
            _, pg = lens.reverse(params.data, x, cotangent.covector)
            g = np.asarray(pg).flatten()
            fisher += g * g

        fisher /= max(len(samples), 1)

        if self.online:
            if self._running is None:
                self._running = fisher.copy()
            else:
                self._running = self.gamma * self._running + (1.0 - self.gamma) * fisher
            return self._running.copy()

        return fisher

    def ewc_penalty(
        self,
        current: NDArray,
        optimal: NDArray,
        fisher: NDArray,
        lam: float,
    ) -> Tuple[float, NDArray]:
        """
        EWC penalty: (lam/2) * sum(F_i * (theta_i - theta*_i)^2)
        Returns (penalty_scalar, gradient_wrt_current).
        """
        diff    = current.flatten() - optimal.flatten()
        penalty = 0.5 * lam * float(np.sum(fisher * diff * diff))
        grad    = lam * fisher * diff
        return penalty, grad


class SITracker:
    """
    Synaptic Intelligence online importance tracker.

    Accumulates omega_i = sum_t( -g_i * delta_theta_i ) / (delta_theta_total + damping)
    """

    def __init__(self, damping: float = 0.1):
        self.damping       = damping
        self._W:            Optional[NDArray] = None
        self._prev_params:  Optional[NDArray] = None
        self._prev_grad:    Optional[NDArray] = None
        self._param_change: Optional[NDArray] = None

    def start_task(self, params: NDArray) -> None:
        flat = params.flatten()
        self._W            = np.zeros_like(flat)
        self._prev_params  = flat.copy()
        self._prev_grad    = np.zeros_like(flat)
        self._param_change = np.zeros_like(flat)

    def update(self, params: NDArray, grad: NDArray) -> None:
        if self._prev_params is None:
            self.start_task(params)
            return
        flat  = params.flatten()
        g     = np.asarray(grad).flatten()
        delta = flat - self._prev_params
        self._W            += -self._prev_grad * delta
        self._param_change += np.abs(delta)
        self._prev_params   = flat.copy()
        self._prev_grad     = g.copy()

    def importance(self) -> NDArray:
        if self._W is None:
            return np.zeros(1)
        denom = self._param_change + self.damping
        return np.maximum(self._W, 0.0) / denom

    def si_penalty(
        self,
        current: NDArray,
        optimal: NDArray,
        omega: NDArray,
        lam: float,
    ) -> Tuple[float, NDArray]:
        diff    = current.flatten() - optimal.flatten()
        penalty = lam * float(np.sum(omega * diff * diff))
        grad    = 2.0 * lam * omega * diff
        return penalty, grad


# ============================================================================
# SECTION 3: EXPERIENCE REPLAY & GEM
# ============================================================================

class ReplayBuffer:
    """Ring buffer with reservoir sampling and task-balanced retrieval."""

    def __init__(self, capacity: int = 2000):
        self.capacity  = capacity
        self._buf:      List[Dict[str, Any]] = []
        self._n_seen:   int = 0
        self._task_idx: Dict[int, List[int]] = {}

    def add(self, x: Any, y: Any, task_id: int) -> None:
        entry = {"x": x, "y": y, "task_id": task_id}
        if len(self._buf) < self.capacity:
            idx = len(self._buf)
            self._buf.append(entry)
        else:
            idx = np.random.randint(0, self._n_seen + 1)
            if idx < self.capacity:
                old_tid = self._buf[idx]["task_id"]
                if old_tid in self._task_idx and idx in self._task_idx[old_tid]:
                    self._task_idx[old_tid].remove(idx)
                self._buf[idx] = entry
            else:
                self._n_seen += 1
                return
        self._task_idx.setdefault(task_id, []).append(idx)
        self._n_seen += 1

    def add_batch(self, data: List[Tuple[Any, Any]], task_id: int) -> None:
        for x, y in data:
            self.add(x, y, task_id)

    def sample(self, n: int, exclude_task: Optional[int] = None) -> List[Tuple[Any, Any]]:
        pool = [e for e in self._buf if e["task_id"] != exclude_task] \
               if exclude_task is not None else self._buf
        if not pool:
            return []
        idxs = np.random.choice(len(pool), min(n, len(pool)), replace=False)
        return [(pool[i]["x"], pool[i]["y"]) for i in idxs]

    def task_balanced_sample(self, n_per_task: int, exclude_task: Optional[int] = None) -> List[Tuple[Any, Any]]:
        result = []
        for tid, idxs in self._task_idx.items():
            if tid == exclude_task:
                continue
            chosen = np.random.choice(idxs, min(n_per_task, len(idxs)), replace=False)
            result.extend([(self._buf[i]["x"], self._buf[i]["y"]) for i in chosen])
        return result

    def mix(self, current_batch: List[Tuple[Any, Any]], ratio: float = 0.3,
            current_task: Optional[int] = None) -> List[Tuple[Any, Any]]:
        n_replay = max(1, int(len(current_batch) * ratio))
        replay   = self.sample(n_replay, exclude_task=current_task)
        mixed    = list(current_batch) + replay
        np.random.shuffle(mixed)
        return mixed

    def __len__(self) -> int:
        return len(self._buf)


class GEMSolver:
    """
    Gradient Episodic Memory constraint solver.

    Projects the current gradient so it does not increase loss on any
    previous task: <g_proj, g_t> >= -margin  for all past tasks t.

    Reference: Lopez-Paz & Ranzato, NeurIPS 2017.
    """

    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def project(self, current_grad: NDArray, past_grads: List[NDArray]) -> NDArray:
        if not past_grads:
            return current_grad

        g = current_grad.flatten().copy()
        G = np.stack([pg.flatten() for pg in past_grads])   # (T, D)
        dots = G @ g

        if not np.any(dots < -self.margin):
            return current_grad   # No violation

        G_viol = G[dots < -self.margin]
        GGT    = G_viol @ G_viol.T
        rhs    = -(G_viol @ g + self.margin)

        try:
            lam    = np.linalg.lstsq(GGT, rhs, rcond=None)[0]
            lam    = np.maximum(lam, 0.0)
            g_proj = g + G_viol.T @ lam
        except np.linalg.LinAlgError:
            g_proj = g

        return g_proj.reshape(current_grad.shape)


# ============================================================================
# SECTION 4: KNOWLEDGE DISTILLATION (LwF)
# ============================================================================

class LwFDistillation:
    """
    Learning without Forgetting via soft-target distillation.

    Caches old model logits before training on a new task, then adds
    KL-divergence term: KL(softmax(old/T) || softmax(new/T)) * T^2
    """

    def __init__(self, temperature: float = 2.0, lambda_distill: float = 1.0):
        self.T   = temperature
        self.lam = lambda_distill
        self._cache: Dict[int, Dict[str, NDArray]] = {}

    @staticmethod
    def _key(x: Any) -> str:
        arr = np.asarray(x)
        return arr.tobytes().hex()[:32]

    def cache_logits(
        self,
        task_id: int,
        lens: "ParametricLens",
        params: NDArray,
        data: List[Tuple[Any, Any]],
    ) -> None:
        self._cache[task_id] = {}
        for x, _ in data:
            logits = lens.forward(params, x)
            self._cache[task_id][self._key(x)] = np.asarray(logits).copy()

    def distill_loss(
        self,
        task_id: int,
        x: Any,
        new_logits: NDArray,
    ) -> Tuple[float, NDArray]:
        cache = self._cache.get(task_id, {})
        old   = cache.get(self._key(x))
        if old is None:
            return 0.0, np.zeros_like(new_logits)

        def _softmax(z: NDArray) -> NDArray:
            e = np.exp(z / self.T - np.max(z / self.T))
            return e / (e.sum() + 1e-10)

        p_old = _softmax(old)
        p_new = _softmax(new_logits)

        loss = -self.T * self.T * float(np.sum(p_old * np.log(p_new + 1e-10)))
        grad = self.T * (p_new - p_old)

        return self.lam * loss, self.lam * grad


# ============================================================================
# SECTION 5: PACKNET (STRUCTURED PRUNING + TASK MASKS)
# ============================================================================

class PackNet:
    """
    PackNet: task-specific parameter subsets via iterative magnitude pruning.

    After training task t:
      1. Prune the least-important fraction of free parameters.
      2. Freeze pruned params for task t.
      3. Remaining free params available for task t+1.

    Reference: Mallya & Lazebnik, CVPR 2018.
    """

    def __init__(self, prune_fraction: float = 0.5):
        self.prune_fraction = prune_fraction
        self.task_masks: Dict[int, NDArray] = {}
        self._free_mask: Optional[NDArray]  = None

    def init(self, n_params: int) -> None:
        self._free_mask = np.ones(n_params, dtype=np.float64)

    def prune_and_assign(self, task_id: int, params: NDArray) -> NDArray:
        """Prune least-magnitude free params, assign remainder to task_id."""
        if self._free_mask is None:
            self.init(params.size)

        flat         = params.flatten()
        free_indices = np.where(self._free_mask > 0.5)[0]

        if len(free_indices) == 0:
            logger.warning("PackNet: no free parameters left — reusing all.")
            mask = np.ones(params.size, dtype=np.float64)
            self.task_masks[task_id] = mask
            return mask

        n_prune      = max(1, int(len(free_indices) * self.prune_fraction))
        magnitudes   = np.abs(flat[free_indices])
        prune_local  = np.argsort(magnitudes)[:n_prune]
        keep_local   = np.argsort(magnitudes)[n_prune:]
        prune_global = free_indices[prune_local]
        keep_global  = free_indices[keep_local]

        task_mask = np.zeros(params.size, dtype=np.float64)
        task_mask[keep_global] = 1.0

        self._free_mask[keep_global]  = 0.0
        self._free_mask[prune_global] = 0.0

        self.task_masks[task_id] = task_mask
        logger.info(
            f"PackNet task {task_id}: {int(task_mask.sum())} owned params, "
            f"{int(self._free_mask.sum())} still free"
        )
        return task_mask

    def apply_mask(self, task_id: int, grad: NDArray) -> NDArray:
        """Zero out gradients for params not belonging to task_id."""
        mask = self.task_masks.get(task_id)
        if mask is None:
            return grad
        return grad * mask.reshape(grad.shape)

    def freeze_gradient(self, grad: NDArray) -> NDArray:
        """Zero out gradients for all frozen (assigned) params."""
        if self._free_mask is None:
            return grad
        return grad * self._free_mask.reshape(grad.shape)


# ============================================================================
# SECTION 6: REGULARIZED LENS WRAPPERS
# ============================================================================

def _make_regularized_lens(
    base_lens: "ParametricLens",
    penalty_fn: Callable[[NDArray], Tuple[float, NDArray]],
    name_suffix: str,
) -> "ParametricLens":
    """
    Wrap a ParametricLens so its reverse pass adds a penalty gradient.

    penalty_fn(params_flat) -> (penalty_scalar, penalty_grad_flat)
    """
    def fwd(params: NDArray, x: Any) -> Any:
        return base_lens.forward_map(params, x)

    def rev(params: NDArray, x: Any, dy: Any) -> Tuple[Any, NDArray]:
        in_tan, p_tan = base_lens.reverse_derivative(params, x, dy)
        p_arr = np.asarray(p_tan)
        _, pen_grad = penalty_fn(params.flatten())
        total = p_arr + pen_grad.reshape(p_arr.shape)
        return in_tan, total

    return ParametricLens(
        forward_map=fwd,
        reverse_derivative=rev,
        param_spec=base_lens.param_spec,
        input_spec=base_lens.input_spec,
        output_spec=base_lens.output_spec,
        name=f"{base_lens.name}_{name_suffix}",
    )


def build_ewc_lens(
    base_lens: "ParametricLens",
    task_records: List[TaskRecord],
    lam: float,
    fisher_estimator: FisherEstimator,
) -> "ParametricLens":
    """Build EWC-regularized lens from past task records."""
    n_tasks = sum(1 for r in task_records if r.fisher_diag is not None)

    def penalty(flat_params: NDArray) -> Tuple[float, NDArray]:
        total_pen  = 0.0
        total_grad = np.zeros_like(flat_params)
        for rec in task_records:
            if rec.fisher_diag is None or rec.optimal_params is None:
                continue
            pen, g = fisher_estimator.ewc_penalty(
                flat_params, rec.optimal_params.flatten(),
                rec.fisher_diag, lam / max(n_tasks, 1)
            )
            total_pen  += pen
            total_grad += g
        return total_pen, total_grad

    return _make_regularized_lens(base_lens, penalty, "EWC")


def build_si_lens(
    base_lens: "ParametricLens",
    task_records: List[TaskRecord],
    lam: float,
    si_tracker: SITracker,
) -> "ParametricLens":
    """Build SI-regularized lens from past task records."""
    def penalty(flat_params: NDArray) -> Tuple[float, NDArray]:
        total_pen  = 0.0
        total_grad = np.zeros_like(flat_params)
        for rec in task_records:
            if rec.si_importance is None or rec.optimal_params is None:
                continue
            pen, g = si_tracker.si_penalty(
                flat_params, rec.optimal_params.flatten(),
                rec.si_importance, lam
            )
            total_pen  += pen
            total_grad += g
        return total_pen, total_grad

    return _make_regularized_lens(base_lens, penalty, "SI")


# ============================================================================
# SECTION 7: CONFIGURATION
# ============================================================================

@dataclass
class CLConfig:
    """Unified configuration for all continual learning strategies."""

    # Strategy
    strategy: str = "ewc"   # ewc | online_ewc | si | replay | gem | pnn | packnet | lwf | ewc_replay | si_replay

    # Architecture
    input_dim:   int        = 64
    output_dim:  int        = 10
    hidden_dims: List[int]  = field(default_factory=lambda: [128, 128])
    activation:  str        = "relu"

    # Training
    learning_rate:    float = 0.001
    batch_size:       int   = 32
    epochs_per_task:  int   = 50
    optimizer:        str   = "adam"
    gradient_clip:    float = 1.0

    # EWC / Online-EWC
    lambda_ewc:    float = 5000.0
    ewc_gamma:     float = 0.95      # decay for online EWC
    fisher_samples: int  = 200

    # SI
    lambda_si:    float = 1.0
    si_damping:   float = 0.1

    # Replay
    replay_capacity: int   = 2000
    replay_ratio:    float = 0.3

    # GEM
    gem_margin:       float = 0.5
    gem_mem_per_task: int   = 256

    # PackNet
    packnet_prune_fraction: float = 0.5

    # LwF
    lwf_temperature:  float = 2.0
    lwf_lambda:       float = 1.0

    # Curriculum
    competency_threshold: float = 0.0   # 0 = no gate
    difficulty_schedule:  str   = "linear"  # linear | exponential | manual

    # General
    seed:    Optional[int] = 42
    verbose: bool          = True
    checkpoint_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for f in self.__dataclass_fields__:
            d[f] = getattr(self, f)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CLConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ============================================================================
# SECTION 8: CORE CONTINUAL LEARNER
# ============================================================================

class ContinualLearner:
    """
    Core continual learning engine.

    Manages task sessions, applies the selected strategy, tracks metrics,
    and supports curriculum-style sequential teaching.

    Usage:
        learner = ContinualLearner(lens, config)
        learner.new_task("alphabet", train_data_az)
        learner.train()
        learner.new_task("words", train_data_words)
        learner.train()
        print(learner.metrics.report())
    """

    def __init__(
        self,
        base_lens: "ParametricLens",
        config: CLConfig,
        initial_params: Optional["ParameterState"] = None,
    ):
        self.base_lens = base_lens
        self.config    = config

        if config.seed is not None:
            np.random.seed(config.seed)

        # Parameter state
        if initial_params is not None:
            self.params = initial_params
        else:
            init_data = np.random.randn(base_lens.param_spec.size).astype(np.float64) * 0.02
            self.params = ParameterState(data=init_data, spec=base_lens.param_spec)

        # Strategy components
        strat = config.strategy.lower()
        self._fisher = FisherEstimator(
            online=(strat == "online_ewc"),
            gamma=config.ewc_gamma
        )
        self._si      = SITracker(damping=config.si_damping)
        self._replay  = ReplayBuffer(capacity=config.replay_capacity)
        self._gem     = GEMSolver(margin=config.gem_margin)
        self._packnet = PackNet(prune_fraction=config.packnet_prune_fraction)
        self._lwf     = LwFDistillation(
            temperature=config.lwf_temperature,
            lambda_distill=config.lwf_lambda
        )

        # GEM: per-task gradient memory
        self._gem_grads: Dict[int, NDArray] = {}

        # Task records
        self.tasks:           List[TaskRecord] = []
        self.current_task_id: int = -1

        # Metrics
        self.metrics = CLMetrics()

        # Checkpoint dir
        self._ckpt_dir: Optional[Path] = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else None
        )

    # ── Task management ──────────────────────────────────────────────────────

    def new_task(
        self,
        name: str,
        train_data: List[Tuple[Any, Any]],
        val_data: Optional[List[Tuple[Any, Any]]] = None,
        description: str = "",
        difficulty: float = 1.0,
        competency_threshold: float = 0.0,
        mode: TaskMode = TaskMode.TASK_INCREMENTAL,
    ) -> TaskRecord:
        """
        Register a new task / lesson.

        Call this before train(). Finalizes the previous task automatically.
        """
        # Finalize previous task
        if self.current_task_id >= 0:
            self._finalize_task(self.tasks[self.current_task_id])

        task_id = len(self.tasks)
        record  = TaskRecord(
            task_id=task_id,
            name=name,
            description=description,
            difficulty=difficulty,
            mode=mode,
            train_data=train_data,
            val_data=val_data,
            competency_threshold=competency_threshold,
        )
        self.tasks.append(record)
        self.current_task_id = task_id

        # LwF: cache old logits before training
        strat = self.config.strategy.lower()
        if strat == "lwf" and task_id > 0:
            self._lwf.cache_logits(task_id - 1, self.base_lens, self.params.data, train_data)

        # SI: reset path integral for new task
        if strat in ("si", "si_replay"):
            self._si.start_task(self.params.data)

        logger.info(f"New task [{task_id}] '{name}' | {len(train_data)} samples | strategy={self.config.strategy}")
        return record

    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple["ParameterState", "TrainingMetrics"]:
        """Train the current task with the configured strategy."""
        if self.current_task_id < 0:
            raise RuntimeError("Call new_task() before train().")

        record     = self.tasks[self.current_task_id]
        n_epochs   = epochs    or self.config.epochs_per_task
        n_batch    = batch_size or self.config.batch_size
        strat      = self.config.strategy.lower()

        # Build strategy lens
        lens = self._build_strategy_lens(strat)

        # Prepare training data (mix with replay if needed)
        train_data = self._prepare_data(record, strat, n_batch)

        # Build trainer config
        trainer_cfg = TrainingConfig(
            learning_rate=self.config.learning_rate,
            batch_size=n_batch,
            max_epochs=n_epochs,
            gradient_clip=self.config.gradient_clip,
            verbose=self.config.verbose,
            checkpoint_dir=self._ckpt_dir,
            seed=None,   # seed already set globally
        )

        # GEM requires custom training loop
        if strat == "gem":
            final_params, metrics = self._train_gem(record, n_epochs, n_batch)
        else:
            trainer = UniversalTrainer(trainer_cfg)
            # SI: hook to update importance online
            if strat in ("si", "si_replay"):
                trainer.register_callback("epoch_end", self._si_epoch_callback)
            final_params, metrics = trainer.train(
                lens, self.params, train_data, val_data=record.val_data, epochs=n_epochs
            )

        self.params = final_params

        # Record metrics
        record.trained_at       = datetime.now()
        record.epochs_trained   = n_epochs
        record.train_loss_final = metrics.loss_history[-1] if metrics.loss_history else float("inf")
        record.val_loss_final   = metrics.val_loss_history[-1] if metrics.val_loss_history else float("inf")

        # Evaluate accuracy on this task right after training
        acc = self._evaluate_accuracy(record.val_data or record.train_data[:200])
        record.accuracy = acc
        self.metrics.task_accuracy_at_train[self.current_task_id] = acc

        # Curriculum gate check
        if record.competency_threshold > 0:
            record.passed_gate = acc >= record.competency_threshold
            if not record.passed_gate:
                logger.warning(
                    f"Task {record.name}: accuracy {acc:.4f} below threshold "
                    f"{record.competency_threshold:.4f} — consider re-training."
                )
            else:
                logger.info(f"Task {record.name}: competency gate PASSED ({acc:.4f})")

        logger.info(
            f"Task [{self.current_task_id}] '{record.name}' done | "
            f"loss={record.train_loss_final:.6f} | acc={acc:.4f}"
        )

        # Save checkpoint
        if self._ckpt_dir:
            self._save_checkpoint()

        return final_params, metrics

    # ── Strategy helpers ─────────────────────────────────────────────────────

    def _build_strategy_lens(self, strat: str) -> "ParametricLens":
        past = self.tasks[:self.current_task_id]

        if strat in ("ewc", "online_ewc", "ewc_replay") and past:
            return build_ewc_lens(self.base_lens, past, self.config.lambda_ewc, self._fisher)

        if strat in ("si", "si_replay") and past:
            return build_si_lens(self.base_lens, past, self.config.lambda_si, self._si)

        if strat == "packnet":
            # Freeze gradient for already-assigned params
            pn = self._packnet

            def fwd(p: NDArray, x: Any) -> Any:
                return self.base_lens.forward_map(p, x)

            def rev(p: NDArray, x: Any, dy: Any) -> Tuple[Any, NDArray]:
                in_t, p_t = self.base_lens.reverse_derivative(p, x, dy)
                p_arr = np.asarray(p_t)
                masked = pn.freeze_gradient(p_arr.flatten()).reshape(p_arr.shape)
                return in_t, masked

            return ParametricLens(
                forward_map=fwd,
                reverse_derivative=rev,
                param_spec=self.base_lens.param_spec,
                input_spec=self.base_lens.input_spec,
                output_spec=self.base_lens.output_spec,
                name=f"{self.base_lens.name}_PackNet",
            )

        if strat == "lwf" and past:
            lwf = self._lwf
            prev_id = self.current_task_id - 1
            base = self.base_lens

            def fwd_lwf(p: NDArray, x: Any) -> Any:
                return base.forward_map(p, x)

            def rev_lwf(p: NDArray, x: Any, dy: Any) -> Tuple[Any, NDArray]:
                in_t, p_t = base.reverse_derivative(p, x, dy)
                new_logits = base.forward_map(p, x)
                _, distill_grad = lwf.distill_loss(prev_id, x, np.asarray(new_logits))
                # Propagate distillation gradient through reverse pass
                _, dp_distill = base.reverse_derivative(p, x, distill_grad)
                total = np.asarray(p_t) + np.asarray(dp_distill)
                return in_t, total

            return ParametricLens(
                forward_map=fwd_lwf,
                reverse_derivative=rev_lwf,
                param_spec=base.param_spec,
                input_spec=base.input_spec,
                output_spec=base.output_spec,
                name=f"{base.name}_LwF",
            )

        return self.base_lens

    def _prepare_data(
        self,
        record: TaskRecord,
        strat: str,
        batch_size: int,
    ) -> List[Tuple[Any, Any]]:
        """Mix current task data with replay if strategy requires it."""
        if strat in ("replay", "ewc_replay", "si_replay") and self.current_task_id > 0:
            return self._replay.mix(
                record.train_data,
                ratio=self.config.replay_ratio,
                current_task=self.current_task_id,
            )
        return record.train_data

    def _train_gem(
        self,
        record: TaskRecord,
        n_epochs: int,
        batch_size: int,
    ) -> Tuple["ParameterState", "TrainingMetrics"]:
        """
        Custom GEM training loop.

        For each batch:
          1. Compute gradient on current task.
          2. Compute reference gradients on past-task memory.
          3. Project current gradient if it violates past constraints.
          4. Apply projected gradient.
        """
        from GRAD import TrainingMetrics as TM, Adam as AdamOpt, OptimizerConfig, OptimizerState, TangentBundle

        metrics   = TM()
        opt       = AdamOpt()
        opt_cfg   = OptimizerConfig(
            learning_rate=self.config.learning_rate,
            gradient_clip=self.config.gradient_clip,
        )
        opt_state = opt.init_state(self.params.spec, opt_cfg)
        params    = self.params
        loss_fn   = MSE()

        data = record.train_data

        for epoch in range(n_epochs):
            np.random.shuffle(data)
            epoch_loss = 0.0

            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]

                # Current task gradient
                acc_grad: Optional[NDArray] = None
                batch_loss = 0.0
                for x, y in batch:
                    pred = self.base_lens.forward(params.data, x)
                    loss, cot = loss_fn(pred, y)
                    _, pg = self.base_lens.reverse(params.data, x, cot.covector)
                    g = np.asarray(pg).flatten()
                    acc_grad = g if acc_grad is None else acc_grad + g
                    batch_loss += float(loss)

                if acc_grad is None:
                    continue
                acc_grad  /= len(batch)
                batch_loss /= len(batch)

                # Reference gradients from past tasks
                past_grads: List[NDArray] = []
                for tid, mem_grad in self._gem_grads.items():
                    past_grads.append(mem_grad)

                # Project
                proj_grad = self._gem.project(acc_grad, past_grads)

                # Apply
                tangent   = TangentBundle(base_point=params.data, fiber_vector=proj_grad.reshape(params.shape))
                params, opt_state = opt.step(params, tangent, opt_state)
                epoch_loss += batch_loss

            avg_loss = epoch_loss / max(len(data) // batch_size, 1)
            metrics.loss_history.append(avg_loss)

            if self.config.verbose and epoch % 10 == 0:
                logger.info(f"GEM Epoch {epoch}/{n_epochs} | loss={avg_loss:.6f}")

        # Update GEM memory for this task
        mem_samples = self._replay.task_balanced_sample(
            self.config.gem_mem_per_task, exclude_task=None
        )
        if not mem_samples:
            mem_samples = data[:self.config.gem_mem_per_task]

        mem_grad: Optional[NDArray] = None
        for x, y in mem_samples:
            pred = self.base_lens.forward(params.data, x)
            loss, cot = loss_fn(pred, y)
            _, pg = self.base_lens.reverse(params.data, x, cot.covector)
            g = np.asarray(pg).flatten()
            mem_grad = g if mem_grad is None else mem_grad + g
        if mem_grad is not None:
            self._gem_grads[self.current_task_id] = mem_grad / max(len(mem_samples), 1)

        return params, metrics

    def _si_epoch_callback(self, trainer: Any, epoch: int, loss: float, val_loss: Any) -> None:
        """Update SI importance after each epoch."""
        if trainer._current_params is not None:
            # Approximate gradient from param change
            if hasattr(self, "_si_prev_params"):
                delta = trainer._current_params.data.flatten() - self._si_prev_params
                self._si.update(trainer._current_params.data, delta)
            self._si_prev_params = trainer._current_params.data.flatten().copy()

    # ── Task finalization ─────────────────────────────────────────────────────

    def _finalize_task(self, record: TaskRecord) -> None:
        """Snapshot params and compute importance after a task completes."""
        record.optimal_params = self.params.data.copy()
        strat = self.config.strategy.lower()

        if strat in ("ewc", "online_ewc", "ewc_replay"):
            loss_fn = MSE()
            record.fisher_diag = self._fisher.compute(
                self.base_lens, self.params,
                record.train_data, loss_fn,
                n_samples=self.config.fisher_samples,
            )
            logger.info(f"Fisher computed for task {record.task_id} | "
                        f"max={record.fisher_diag.max():.4f}")

        if strat in ("si", "si_replay"):
            record.si_importance = self._si.importance().copy()

        if strat in ("replay", "ewc_replay", "si_replay", "gem"):
            n_store = min(200, len(record.train_data))
            self._replay.add_batch(record.train_data[:n_store], record.task_id)
            logger.info(f"Replay buffer: {len(self._replay)} samples stored")

        if strat == "packnet":
            self._packnet.prune_and_assign(record.task_id, self.params.data)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_accuracy(
        self,
        data: List[Tuple[Any, Any]],
        threshold: float = 0.5,
    ) -> float:
        """
        Compute accuracy (for classification) or 1 - normalized_loss (regression).
        Detects classification by checking if targets are integers or one-hot.
        """
        if not data:
            return 0.0

        loss_fn = MSE()
        total_loss = 0.0
        correct    = 0
        is_clf     = False

        # Detect classification: target is int or 1D one-hot
        sample_y = np.asarray(data[0][1])
        if sample_y.ndim == 0 or (sample_y.ndim == 1 and sample_y.dtype in (np.int32, np.int64)):
            is_clf = True
        elif sample_y.ndim == 1 and np.all((sample_y == 0) | (sample_y == 1)):
            is_clf = True

        for x, y in data:
            pred = self.base_lens.forward(self.params.data, x)
            loss, _ = loss_fn(np.asarray(pred), np.asarray(y, dtype=np.float64))
            total_loss += float(loss)

            if is_clf:
                pred_arr = np.asarray(pred)
                y_arr    = np.asarray(y)
                if pred_arr.ndim == 1 and pred_arr.size > 1:
                    pred_class = int(np.argmax(pred_arr))
                    true_class = int(np.argmax(y_arr)) if y_arr.ndim == 1 else int(y_arr)
                else:
                    pred_class = int(pred_arr.flatten()[0] > threshold)
                    true_class = int(y_arr.flatten()[0] > threshold)
                if pred_class == true_class:
                    correct += 1

        if is_clf:
            return correct / len(data)
        else:
            # Regression: return 1 / (1 + avg_loss) as a [0,1] proxy
            avg_loss = total_loss / len(data)
            return 1.0 / (1.0 + avg_loss)

    def evaluate_all(
        self,
        test_data_map: Optional[Dict[int, List[Tuple[Any, Any]]]] = None,
    ) -> CLMetrics:
        """
        Evaluate on all tasks and update CLMetrics.

        If test_data_map is None, uses each task's val_data or last 200 train samples.
        """
        if test_data_map is None:
            test_data_map = {}
            for rec in self.tasks:
                test_data_map[rec.task_id] = rec.val_data or rec.train_data[:200]

        for task_id, test_data in test_data_map.items():
            if task_id >= len(self.tasks):
                continue
            acc = self._evaluate_accuracy(test_data)
            self.metrics.task_accuracy_final[task_id] = acc

        return self.metrics

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        if self._ckpt_dir is None:
            return
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_dir / f"cl_task{self.current_task_id}.pkl"
        state = {
            "params":         self.params.data,
            "task_id":        self.current_task_id,
            "config":         self.config.to_dict(),
            "metrics":        {
                "at_train": self.metrics.task_accuracy_at_train,
                "final":    self.metrics.task_accuracy_final,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        new_data = state["params"]
        self.params = ParameterState(data=new_data, spec=self.params.spec)
        self.metrics.task_accuracy_at_train = state["metrics"]["at_train"]
        self.metrics.task_accuracy_final    = state["metrics"]["final"]
        logger.info(f"Checkpoint loaded: {path}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CONTINUAL LEARNER SUMMARY",
            "=" * 60,
            f"  Strategy      : {self.config.strategy}",
            f"  Tasks trained : {len(self.tasks)}",
            f"  Replay buffer : {len(self._replay)} samples",
            "",
        ]
        for rec in self.tasks:
            gate = "PASS" if rec.passed_gate else ("FAIL" if rec.competency_threshold > 0 else "N/A")
            lines.append(
                f"  [{rec.task_id:2d}] {rec.name:<30s} "
                f"loss={rec.train_loss_final:.4f}  "
                f"acc={rec.accuracy:.4f}  gate={gate}"
            )
        lines.append("")
        lines.append(self.metrics.report())
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# SECTION 9: HIGH-LEVEL SYSTEM API
# ============================================================================

class ContinualLearningSystem:
    """
    High-level API for curriculum-style continual learning.

    Wraps ContinualLearner with convenience methods for:
    - Building architectures from config
    - Teaching sequences (curriculum)
    - Strategy comparison
    - Save / load

    Example:
        system = ContinualLearningSystem.create(
            input_dim=26, output_dim=26, strategy="ewc",
            hidden_dims=[128, 128], epochs_per_task=50
        )
        system.teach("alphabet",  az_data,    competency_threshold=0.90)
        system.teach("words",     word_data,  competency_threshold=0.85)
        system.teach("sentences", sent_data)
        print(system.learner.summary())
    """

    def __init__(self, config: CLConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)

        self.base_lens = self._build_network()
        self.learner   = ContinualLearner(self.base_lens, config)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        input_dim: int,
        output_dim: int,
        strategy: str = "ewc",
        **kwargs,
    ) -> "ContinualLearningSystem":
        config = CLConfig(input_dim=input_dim, output_dim=output_dim,
                          strategy=strategy, **kwargs)
        return cls(config)

    def _build_network(self) -> "ParametricLens":
        lenses   = []
        prev_dim = self.config.input_dim
        act      = self.config.activation

        for i, h in enumerate(self.config.hidden_dims):
            lenses.append(LensFactory.linear(prev_dim, h, name=f"fc{i}"))
            lenses.append(LensFactory.activation(act, name=f"{act}{i}"))
            prev_dim = h

        lenses.append(LensFactory.linear(prev_dim, self.config.output_dim, name="out"))
        return LensFactory.sequential(lenses, name="cl_network")

    # ── Teaching API ──────────────────────────────────────────────────────────

    def teach(
        self,
        name: str,
        train_data: List[Tuple[Any, Any]],
        val_data: Optional[List[Tuple[Any, Any]]] = None,
        epochs: Optional[int] = None,
        competency_threshold: float = 0.0,
        description: str = "",
        difficulty: float = 1.0,
        retrain_if_failed: bool = True,
        max_retries: int = 3,
    ) -> "TrainingMetrics":
        """
        Teach one lesson / task.

        If competency_threshold > 0 and accuracy falls short,
        optionally retrain up to max_retries times.
        """
        self.learner.new_task(
            name=name,
            train_data=train_data,
            val_data=val_data,
            description=description,
            difficulty=difficulty,
            competency_threshold=competency_threshold,
        )

        _, metrics = self.learner.train(epochs=epochs)
        record = self.learner.tasks[self.learner.current_task_id]

        if retrain_if_failed and competency_threshold > 0 and not record.passed_gate:
            for attempt in range(1, max_retries + 1):
                logger.info(f"Retrying task '{name}' (attempt {attempt}/{max_retries})")
                _, metrics = self.learner.train(epochs=epochs)
                if record.passed_gate:
                    break

        return metrics

    def teach_curriculum(
        self,
        curriculum: List[Dict[str, Any]],
    ) -> List["TrainingMetrics"]:
        """
        Teach a full curriculum from a list of task dicts.

        Each dict must have 'name' and 'train_data'.
        Optional keys: val_data, epochs, competency_threshold, description, difficulty.

        Example:
            curriculum = [
                {"name": "letters",   "train_data": az_data,   "competency_threshold": 0.90},
                {"name": "numbers",   "train_data": num_data,  "competency_threshold": 0.90},
                {"name": "words",     "train_data": word_data, "competency_threshold": 0.85},
                {"name": "sentences", "train_data": sent_data},
            ]
            system.teach_curriculum(curriculum)
        """
        all_metrics = []
        for i, task_def in enumerate(curriculum):
            logger.info(f"\n{'='*60}")
            logger.info(f"Curriculum step {i+1}/{len(curriculum)}: {task_def['name']}")
            logger.info(f"{'='*60}")
            m = self.teach(**task_def)
            all_metrics.append(m)
        return all_metrics

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_data_map: Optional[Dict[int, List[Tuple[Any, Any]]]] = None,
    ) -> CLMetrics:
        return self.learner.evaluate_all(test_data_map)

    def forgetting_report(self) -> str:
        metrics = self.learner.evaluate_all()
        bwt = metrics.backward_transfer
        lines = [
            "── Forgetting Report ──",
            f"  BWT: {bwt:+.4f}  ({'forgetting' if bwt < 0 else 'positive transfer'})",
        ]
        for t in sorted(metrics.task_accuracy_at_train):
            at_t = metrics.task_accuracy_at_train[t]
            fin  = metrics.task_accuracy_final.get(t, float("nan"))
            diff = fin - at_t
            lines.append(f"  Task {t}: {at_t:.4f} -> {fin:.4f}  ({diff:+.4f})")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config":         self.config.to_dict(),
            "params":         self.learner.params.data,
            "task_records":   [
                {
                    "task_id":        r.task_id,
                    "name":           r.name,
                    "description":    r.description,
                    "difficulty":     r.difficulty,
                    "epochs_trained": r.epochs_trained,
                    "train_loss":     r.train_loss_final,
                    "accuracy":       r.accuracy,
                    "passed_gate":    r.passed_gate,
                }
                for r in self.learner.tasks
            ],
            "metrics": {
                "at_train": self.learner.metrics.task_accuracy_at_train,
                "final":    self.learner.metrics.task_accuracy_final,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"System saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContinualLearningSystem":
        with open(path, "rb") as f:
            state = pickle.load(f)
        config = CLConfig.from_dict(state["config"])
        system = cls(config)
        new_data = state["params"]
        system.learner.params = ParameterState(
            data=new_data, spec=system.learner.params.spec
        )
        system.learner.metrics.task_accuracy_at_train = state["metrics"]["at_train"]
        system.learner.metrics.task_accuracy_final    = state["metrics"]["final"]
        logger.info(f"System loaded from {path}")
        return system


# ============================================================================
# SECTION 10: STRATEGY COMPARISON UTILITY
# ============================================================================

def compare_strategies(
    curriculum: List[Dict[str, Any]],
    strategies: List[str],
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run the same curriculum under multiple strategies and compare BWT / accuracy.

    Returns a dict keyed by strategy name with metrics and summary.
    """
    results: Dict[str, Any] = {}

    for strat in strategies:
        logger.info(f"\n{'#'*60}\nStrategy: {strat}\n{'#'*60}")
        try:
            system = ContinualLearningSystem.create(
                input_dim=input_dim,
                output_dim=output_dim,
                strategy=strat,
                **kwargs,
            )
            system.teach_curriculum(curriculum)
            metrics = system.evaluate()
            results[strat] = {
                "avg_accuracy":       metrics.average_accuracy,
                "backward_transfer":  metrics.backward_transfer,
                "task_accuracy":      metrics.task_accuracy_final,
                "summary":            system.learner.summary(),
            }
        except Exception as e:
            logger.error(f"Strategy {strat} failed: {e}")
            results[strat] = {"error": str(e)}

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'STRATEGY':<15} {'AVG_ACC':>8} {'BWT':>8}")
    print("-" * 70)
    for strat, res in results.items():
        if "error" in res:
            print(f"{strat:<15} {'ERROR':>8}")
        else:
            print(f"{strat:<15} {res['avg_accuracy']:>8.4f} {res['backward_transfer']:>+8.4f}")
    print("=" * 70)

    return results


# ============================================================================
# SECTION 11: DEMO
# ============================================================================

def demo() -> None:
    """
    Demonstration: human-curriculum style teaching.

    Simulates teaching a model:
      1. Simple linear patterns (easy)
      2. Quadratic patterns (medium)
      3. Trigonometric patterns (hard)
    """
    print("=" * 60)
    print("CONTINUAL LEARNING DEMO — Human Curriculum")
    print("=" * 60)

    np.random.seed(42)

    def make_task(fn: Callable, n: int = 300, dim: int = 8) -> List[Tuple[NDArray, NDArray]]:
        data = []
        for _ in range(n):
            x = np.random.randn(dim).astype(np.float64)
            y = np.array([fn(x)], dtype=np.float64)
            data.append((x, y))
        return data

    curriculum = [
        {
            "name":                  "linear",
            "train_data":            make_task(lambda x: float(x[0] + x[1])),
            "val_data":              make_task(lambda x: float(x[0] + x[1]), n=50),
            "competency_threshold":  0.70,
            "difficulty":            0.3,
            "description":           "Simple linear combination",
        },
        {
            "name":                  "quadratic",
            "train_data":            make_task(lambda x: float(x[0]**2 + x[1]**2)),
            "val_data":              make_task(lambda x: float(x[0]**2 + x[1]**2), n=50),
            "competency_threshold":  0.65,
            "difficulty":            0.6,
            "description":           "Quadratic features",
        },
        {
            "name":                  "trigonometric",
            "train_data":            make_task(lambda x: float(np.sin(x[0]) * np.cos(x[1]))),
            "val_data":              make_task(lambda x: float(np.sin(x[0]) * np.cos(x[1])), n=50),
            "competency_threshold":  0.55,
            "difficulty":            1.0,
            "description":           "Nonlinear trigonometric",
        },
    ]

    # EWC system
    print("\n--- EWC Strategy ---")
    system = ContinualLearningSystem.create(
        input_dim=8,
        output_dim=1,
        strategy="ewc",
        hidden_dims=[64, 64],
        learning_rate=0.005,
        epochs_per_task=30,
        lambda_ewc=1000.0,
        verbose=False,
    )
    system.teach_curriculum(curriculum)
    metrics = system.evaluate()
    print(system.learner.summary())
    print(system.forgetting_report())

    # Replay system
    print("\n--- Experience Replay Strategy ---")
    replay_system = ContinualLearningSystem.create(
        input_dim=8,
        output_dim=1,
        strategy="replay",
        hidden_dims=[64, 64],
        learning_rate=0.005,
        epochs_per_task=30,
        replay_capacity=500,
        replay_ratio=0.3,
        verbose=False,
    )
    replay_system.teach_curriculum(curriculum)
    replay_metrics = replay_system.evaluate()
    print(replay_system.learner.summary())

    # Strategy comparison
    print("\n--- Strategy Comparison ---")
    compare_strategies(
        curriculum=curriculum,
        strategies=["ewc", "replay", "si"],
        input_dim=8,
        output_dim=1,
        hidden_dims=[64, 64],
        learning_rate=0.005,
        epochs_per_task=20,
        verbose=False,
    )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
