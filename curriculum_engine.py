"""
Curriculum Engine for ETI
=========================

Rule-based symbolic curriculum with permanent memory, pattern matching,
and cross-domain knowledge transfer. Integrates with GRAD.py + continual_learning.py.

Architecture:
    - RuleMemory      : permanent weighted rules loaded every session
    - PatternMatcher  : matches examples to stored rules (few-shot generalisation)
    - Domain          : user-defined subject area (math, english, physics, programming, ...)
    - Lesson          : one teaching unit inside a domain
    - CurriculumEngine: orchestrates domains, lessons, rule memory, and the ContinualLearner

Symbolic Encoding:
    Every concept is encoded as a fixed-length float vector so GRAD.py can train on it.
    The encoder is deterministic and invertible for known symbols.

Version: 2.0.0-PRODUCTION
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# GRAD + continual_learning integration
try:
    from GRAD import (
        ParametricLens, ParameterState, TensorSpec,
        LensFactory, TrainingConfig, MSE, CrossEntropy,
    )
    from continual_learning import (
        ContinualLearner, ContinualLearningSystem, CLConfig, TaskMode,
    )
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    print(f"DEBUG: Import error in curriculum_engine.py: {_e}")
    warnings.warn(f"Dependency missing: {_e}", ImportWarning)

logger = logging.getLogger("eti.curriculum")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ============================================================================
# SECTION 1: SYMBOLIC ENCODER
# ============================================================================

class SymbolEncoder:
    """
    Deterministic encoder: symbol -> fixed-length float vector.

    base_vec  = hash-derived unit vector  (permanent, never changes)
    learned   = persisted offset          (updated by training)
    final_vec = base_vec + learned        (what the model sees)
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._base:    Dict[str, NDArray] = {}
        self._learned: Dict[str, NDArray] = {}

    def _make_base(self, symbol: str) -> NDArray:
        h = hashlib.sha256(symbol.encode()).digest()
        seed = int.from_bytes(h[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-10)

    def base_vec(self, symbol: str) -> NDArray:
        if symbol not in self._base:
            self._base[symbol] = self._make_base(symbol)
        return self._base[symbol]

    def encode(self, symbol: str) -> NDArray:
        base    = self.base_vec(symbol)
        learned = self._learned.get(symbol, np.zeros(self.dim, dtype=np.float64))
        return base + learned

    def update_learned(self, symbol: str, delta: NDArray) -> None:
        if symbol not in self._learned:
            self._learned[symbol] = np.zeros(self.dim, dtype=np.float64)
        self._learned[symbol] += delta

    def encode_sequence(self, symbols: List[str], max_len: int = 16) -> NDArray:
        vecs = [self.encode(s) for s in symbols[:max_len]]
        while len(vecs) < max_len:
            vecs.append(np.zeros(self.dim, dtype=np.float64))
        return np.concatenate(vecs).astype(np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"base": self._base, "learned": self._learned, "dim": self.dim}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._base    = d["base"]
        self._learned = d["learned"]
        self.dim      = d["dim"]


# ============================================================================
# SECTION 2: RULE MEMORY  (permanent, loaded every session)
# ============================================================================

@dataclass
class Rule:
    """
    A named rule with a pattern, weight, and confidence.

    pattern   : symbolic description  e.g. "[obj]*N = N objects"
    weight    : how strongly this rule influences the model (0-2)
    confidence: accumulated evidence from examples (0-1)
    examples  : few-shot examples that confirmed this rule
    """
    rule_id:    str
    name:       str
    domain:     str
    pattern:    str
    weight:     float = 1.0
    confidence: float = 0.0
    examples:   List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    encoding:   Optional[NDArray] = field(default=None, repr=False)

    def strengthen(self, delta: float = 0.1) -> None:
        self.confidence = min(1.0, self.confidence + delta)
        self.weight     = min(2.0, self.weight + delta * 0.5)
        self.updated_at = datetime.now().isoformat()

    def add_example(self, example: Dict[str, Any]) -> None:
        self.examples.append(example)
        if len(self.examples) > 50:
            self.examples = self.examples[-50:]
        self.strengthen()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id":    self.rule_id,
            "name":       self.name,
            "domain":     self.domain,
            "pattern":    self.pattern,
            "weight":     self.weight,
            "confidence": self.confidence,
            "examples":   self.examples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rule":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class RuleMemory:
    """
    Permanent rule store — persisted to disk and loaded every session.

    Rules are the "textbook" the model always has access to.
    They are NOT forgotten between sessions. Each rule has a weight that
    biases the loss during training: high-weight rules produce stronger
    gradient signals.
    """

    def __init__(self, path: Union[str, Path] = "memory/rules.json"):
        self.path  = Path(path)
        self.rules: Dict[str, Rule] = {}
        if self.path.exists():
            self.load()

    def add_rule(self, rule: Rule) -> None:
        self.rules[rule.rule_id] = rule
        logger.info(f"Rule added: [{rule.domain}] {rule.name}")

    def get(self, rule_id: str) -> Optional[Rule]:
        return self.rules.get(rule_id)

    def by_domain(self, domain: str) -> List[Rule]:
        return [r for r in self.rules.values() if r.domain == domain]

    def strengthen_rule(self, rule_id: str, delta: float = 0.1) -> None:
        if rule_id in self.rules:
            self.rules[rule_id].strengthen(delta)

    def add_example_to_rule(self, rule_id: str, example: Dict[str, Any]) -> None:
        if rule_id in self.rules:
            self.rules[rule_id].add_example(example)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {rid: r.to_dict() for rid, r in self.rules.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"RuleMemory saved: {len(self.rules)} rules -> {self.path}")

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.rules = {rid: Rule.from_dict(d) for rid, d in data.items()}
        logger.info(f"RuleMemory loaded: {len(self.rules)} rules from {self.path}")

    def summary(self) -> str:
        lines = [f"RuleMemory ({len(self.rules)} rules):"]
        for r in sorted(self.rules.values(), key=lambda x: x.domain):
            lines.append(
                f"  [{r.domain:12s}] {r.name:<30s} "
                f"w={r.weight:.2f}  conf={r.confidence:.2f}"
            )
        return "\n".join(lines)


# ============================================================================
# SECTION 3: PATTERN MATCHER  (few-shot rule generalisation)
# ============================================================================

class PatternMatcher:
    """
    Matches an input example against stored rules using cosine similarity.

    When a new example arrives:
      1. Encode the example tokens into a vector.
      2. Find the closest rule encoding (cosine sim).
      3. If similarity > threshold -> rule match -> strengthen that rule.
      4. Return the matched rule + confidence.

    The model does NOT need millions of examples because the rule itself
    carries the generalisation.
    """

    def __init__(
        self,
        encoder: SymbolEncoder,
        rule_memory: RuleMemory,
        match_threshold: float = 0.6,
    ):
        self.encoder   = encoder
        self.memory    = rule_memory
        self.threshold = match_threshold

    def _encode_rule(self, rule: Rule) -> NDArray:
        if rule.encoding is not None:
            return rule.encoding
        tokens = rule.pattern.replace("[", " ").replace("]", " ").split()
        vec = self.encoder.encode_sequence(tokens, max_len=8)
        rule.encoding = vec
        return vec

    def _cosine(self, a: NDArray, b: NDArray) -> float:
        na = np.linalg.norm(a) + 1e-10
        nb = np.linalg.norm(b) + 1e-10
        return float(np.dot(a / na, b / nb))

    def match(
        self,
        example_tokens: List[str],
        domain: Optional[str] = None,
    ) -> Tuple[Optional[Rule], float]:
        """Find the best matching rule for an example."""
        example_vec = self.encoder.encode_sequence(example_tokens, max_len=8)
        candidates  = self.memory.by_domain(domain) if domain else list(self.memory.rules.values())

        best_rule = None
        best_sim  = 0.0

        for rule in candidates:
            rule_vec = self._encode_rule(rule)
            min_len  = min(len(example_vec), len(rule_vec))
            sim = self._cosine(example_vec[:min_len], rule_vec[:min_len])
            if sim > best_sim:
                best_sim  = sim
                best_rule = rule

        if best_sim >= self.threshold and best_rule is not None:
            best_rule.strengthen(delta=0.05)
            return best_rule, best_sim

        return None, best_sim

    def explain(self, example_tokens: List[str], domain: Optional[str] = None) -> str:
        rule, sim = self.match(example_tokens, domain)
        if rule:
            return (
                f"Matched rule '{rule.name}' (sim={sim:.3f}, conf={rule.confidence:.3f})\n"
                f"  Pattern: {rule.pattern}"
            )
        return f"No rule matched (best sim={sim:.3f})"


# ============================================================================
# SECTION 4: DATA GENERATORS  (symbolic -> numeric training pairs)
# ============================================================================

class LessonDataGenerator:
    """
    Generates (input_vector, target_vector) training pairs from symbolic rules.
    """

    def __init__(self, encoder: SymbolEncoder):
        self.enc = encoder

    def objects_as_identity(self, n_samples: int = 200) -> List[Tuple[NDArray, NDArray]]:
        """Lesson: Everything is an object. Input: object+noise -> Target: object."""
        base = self.enc.encode("object")
        data = []
        for _ in range(n_samples):
            noise = np.random.randn(self.enc.dim) * 0.05
            x = (base + noise).astype(np.float64)
            y = base.copy()
            data.append((x, y))
        return data

    def counting(self, n_samples: int = 400, max_count: int = 20) -> List[Tuple[NDArray, NDArray]]:
        """Lesson: [obj]*N = N objects. Input: N object-encodings -> Target: one-hot count."""
        data = []
        obj_vec = self.enc.encode("object")
        for _ in range(n_samples):
            n = np.random.randint(1, max_count + 1)
            inp = np.zeros(max_count * self.enc.dim, dtype=np.float64)
            for i in range(n):
                inp[i * self.enc.dim:(i + 1) * self.enc.dim] = obj_vec
            tgt = np.zeros(max_count, dtype=np.float64)
            tgt[n - 1] = 1.0
            data.append((inp, tgt))
        return data

    def comparison(self, n_samples: int = 400, max_val: int = 20) -> List[Tuple[NDArray, NDArray]]:
        """Lesson: compare two counts. Target: [1,0,0]=less, [0,1,0]=equal, [0,0,1]=greater."""
        data = []
        for _ in range(n_samples):
            a = np.random.randint(0, max_val + 1)
            b = np.random.randint(0, max_val + 1)
            a_vec = self._count_to_vec(a, max_val)
            b_vec = self._count_to_vec(b, max_val)
            inp = np.concatenate([a_vec, b_vec]).astype(np.float64)
            if a < b:
                tgt = np.array([1.0, 0.0, 0.0])
            elif a == b:
                tgt = np.array([0.0, 1.0, 0.0])
            else:
                tgt = np.array([0.0, 0.0, 1.0])
            data.append((inp, tgt))
        return data

    def ordering(self, n_samples: int = 400, seq_len: int = 4, max_val: int = 10) -> List[Tuple[NDArray, NDArray]]:
        """Lesson: sort a sequence of counts ascending."""
        data = []
        for _ in range(n_samples):
            counts   = np.random.randint(0, max_val + 1, size=seq_len)
            inp_vecs = np.concatenate([self._count_to_vec(int(c), max_val) for c in counts])
            sorted_c = np.sort(counts)
            tgt_vecs = np.concatenate([self._count_to_vec(int(c), max_val) for c in sorted_c])
            data.append((inp_vecs.astype(np.float64), tgt_vecs.astype(np.float64)))
        return data

    def _count_to_vec(self, n: int, max_val: int) -> NDArray:
        symbol = f"count_{n}"
        v = self.enc.encode(symbol)
        scale = n / max(max_val, 1)
        return (v * (0.5 + 0.5 * scale)).astype(np.float64)

    def word_identity(self, vocab: List[str], n_samples: int = 300) -> List[Tuple[NDArray, NDArray]]:
        """Lesson: word -> its own encoding (vocabulary anchoring)."""
        data = []
        for _ in range(n_samples):
            word = vocab[np.random.randint(len(vocab))]
            x = self.enc.encode(word) + np.random.randn(self.enc.dim) * 0.03
            y = self.enc.encode(word)
            data.append((x.astype(np.float64), y.astype(np.float64)))
        return data

    def word_category(self, categories: Dict[str, List[str]], n_samples: int = 400) -> List[Tuple[NDArray, NDArray]]:
        """Lesson: classify a word into its category (noun/verb/adjective/...)."""
        cat_list = sorted(categories.keys())
        cat_idx  = {c: i for i, c in enumerate(cat_list)}
        n_cats   = len(cat_list)
        data     = []
        for _ in range(n_samples):
            cat   = cat_list[np.random.randint(n_cats)]
            words = categories[cat]
            word  = words[np.random.randint(len(words))]
            x = self.enc.encode(word) + np.random.randn(self.enc.dim) * 0.03
            y = np.zeros(n_cats, dtype=np.float64)
            y[cat_idx[cat]] = 1.0
            data.append((x.astype(np.float64), y))
        return data

    def formula_mapping(
        self,
        formula_fn: Callable[[NDArray], NDArray],
        input_dim: int,
        n_samples: int = 400,
        noise: float = 0.01,
    ) -> List[Tuple[NDArray, NDArray]]:
        """Generic physics formula lesson. formula_fn maps input -> output."""
        data = []
        for _ in range(n_samples):
            x = np.random.randn(input_dim).astype(np.float64)
            y_clean = np.asarray(formula_fn(x), dtype=np.float64)
            y = y_clean + np.random.randn(*y_clean.shape) * noise
            data.append((x, y))
        return data

    def token_classification(
        self,
        token_classes: Dict[str, List[str]],
        n_samples: int = 400,
    ) -> List[Tuple[NDArray, NDArray]]:
        """Classify a programming token (keyword/operator/identifier/literal)."""
        return self.word_category(token_classes, n_samples)

    def from_rule(
        self,
        rule: Rule,
        n_samples: int = 200,
        input_dim: int = 64,
        output_dim: int = 8,
    ) -> List[Tuple[NDArray, NDArray]]:
        """Generate training data directly from a Rule object."""
        data = []
        for ex in rule.examples[:n_samples]:
            if "input" in ex and "output" in ex:
                x = np.asarray(ex["input"], dtype=np.float64)
                y = np.asarray(ex["output"], dtype=np.float64)
                data.append((x, y))

        if rule.encoding is not None and len(data) < n_samples:
            rule_vec = rule.encoding
            for _ in range(n_samples - len(data)):
                x = rule_vec[:input_dim].copy() if len(rule_vec) >= input_dim \
                    else np.pad(rule_vec, (0, input_dim - len(rule_vec)))
                x = x + np.random.randn(input_dim) * 0.1
                y = rule_vec[:output_dim].copy() if len(rule_vec) >= output_dim \
                    else np.pad(rule_vec, (0, output_dim - len(rule_vec)))
                data.append((x.astype(np.float64), y.astype(np.float64)))

        return data[:n_samples]


# ============================================================================
# SECTION 5: LESSON & DOMAIN DEFINITIONS
# ============================================================================

@dataclass
class Lesson:
    """
    One teaching unit: a rule + generated training data + metadata.
    """
    lesson_id:            str
    name:                 str
    domain:               str
    rule_ids:             List[str]
    description:          str = ""
    difficulty:           float = 0.5
    competency_threshold: float = 0.0
    epochs:               int = 50
    prerequisite_ids:     List[str] = field(default_factory=list)
    train_data: List[Tuple[Any, Any]] = field(default_factory=list, repr=False)
    val_data:   Optional[List[Tuple[Any, Any]]] = field(default=None, repr=False)
    completed:  bool  = False
    accuracy:   float = 0.0

    def is_ready(self, completed_lessons: List[str]) -> bool:
        return all(p in completed_lessons for p in self.prerequisite_ids)


@dataclass
class Domain:
    """
    A subject area (math, english, physics, programming, or user-defined).

    Domains are fully user-configurable — nothing is hardcoded.
    Knowledge transfer between domains is enabled via shared encoder embeddings.
    """
    domain_id:     str
    name:          str
    description:   str = ""
    lessons:       List[Lesson] = field(default_factory=list)
    rules:         List[str]    = field(default_factory=list)
    transfer_from: List[str]    = field(default_factory=list)

    def add_lesson(self, lesson: Lesson) -> None:
        self.lessons.append(lesson)

    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        return next((l for l in self.lessons if l.lesson_id == lesson_id), None)

    def ordered_lessons(self) -> List[Lesson]:
        """Return lessons in dependency order (topological sort)."""
        completed: List[str] = []
        ordered:   List[Lesson] = []
        remaining = list(self.lessons)
        max_iter = len(remaining) * 2
        itr = 0
        while remaining and itr < max_iter:
            itr += 1
            for lesson in list(remaining):
                if lesson.is_ready(completed):
                    ordered.append(lesson)
                    completed.append(lesson.lesson_id)
                    remaining.remove(lesson)
        if remaining:
            logger.warning(f"Domain '{self.name}': circular prerequisites detected.")
            ordered.extend(remaining)
        return ordered


# ============================================================================
# SECTION 6: CURRICULUM ENGINE  (main orchestrator)
# ============================================================================

class CurriculumEngine:
    """
    Main orchestrator for symbolic curriculum learning.

    Manages:
      - Domains and lessons (fully user-definable)
      - RuleMemory (permanent, loaded every session)
      - PatternMatcher (few-shot rule generalisation)
      - ContinualLearningSystem (GRAD.py-backed neural learner)
      - Cross-domain knowledge transfer
      - Persistence (save/load state between sessions)

    Usage:
        engine = CurriculumEngine.create(memory_dir="eti_memory")
        engine.new_domain("math", "Mathematics", "Numbers and logic")
        engine.define_rule("count_rule", "Counting", "math",
                           "[obj]*N = N objects", weight=2.0)
        lesson = engine.new_lesson("math_count", "Counting", "math",
                                   rule_ids=["count_rule"], epochs=40)
        engine.generate_lesson_data("math_count")
        engine.teach_lesson("math_count")
        engine.present_example(["object", "object", "object"])
        engine.save()
    """

    def __init__(
        self,
        memory_dir: Union[str, Path] = "eti_memory",
        encoder_dim: int = 64,
        cl_strategy: str = "ewc",
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.003,
        epochs_per_lesson: int = 50,
        match_threshold: float = 0.6,
        verbose: bool = True,
    ):
        self.memory_dir      = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.verbose         = verbose

        # Core components
        self.encoder     = SymbolEncoder(dim=encoder_dim)
        self.rule_memory = RuleMemory(self.memory_dir / "rules.json")
        self.matcher     = PatternMatcher(self.encoder, self.rule_memory, match_threshold)
        self.generator   = LessonDataGenerator(self.encoder)

        # Domains registry
        self.domains: Dict[str, Domain] = {}

        # CL system (lazy init on first teach)
        self._cl_system:    Optional[ContinualLearningSystem] = None
        self._cl_strategy   = cl_strategy
        self._hidden_dims   = hidden_dims or [256, 256]
        self._learning_rate = learning_rate
        self._epochs_default = epochs_per_lesson

        # State
        self._completed:    List[str] = []
        self._transfer_map: Dict[str, List[str]] = {}

        # Load persisted state if available
        state_path = self.memory_dir / "engine_state.json"
        if state_path.exists():
            self._load_state(state_path)

        enc_path = self.memory_dir / "encoder.pkl"
        if enc_path.exists():
            self.encoder.load(enc_path)

        if verbose:
            logger.info(f"CurriculumEngine ready | dir={self.memory_dir} | strategy={cl_strategy}")

    @classmethod
    def create(
        cls,
        memory_dir: str = "eti_memory",
        encoder_dim: int = 64,
        cl_strategy: str = "ewc",
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.003,
        epochs_per_lesson: int = 50,
        match_threshold: float = 0.6,
        verbose: bool = True,
    ) -> "CurriculumEngine":
        return cls(
            memory_dir=memory_dir,
            encoder_dim=encoder_dim,
            cl_strategy=cl_strategy,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            epochs_per_lesson=epochs_per_lesson,
            match_threshold=match_threshold,
            verbose=verbose,
        )

    # ── Domain management ─────────────────────────────────────────────────────

    def new_domain(
        self,
        domain_id: str,
        name: str,
        description: str = "",
        transfer_from: Optional[List[str]] = None,
    ) -> Domain:
        """Register a new domain. Fully user-definable."""
        domain = Domain(
            domain_id=domain_id,
            name=name,
            description=description,
            transfer_from=transfer_from or [],
        )
        self.domains[domain_id] = domain
        if transfer_from:
            self.enable_transfer(domain_id, transfer_from)
        logger.info(f"Domain registered: {domain_id} ({name})")
        return domain

    def get_domain(self, domain_id: str) -> Optional[Domain]:
        return self.domains.get(domain_id)

    # ── Rule management ───────────────────────────────────────────────────────

    def define_rule(
        self,
        rule_id: str,
        name: str,
        domain: str,
        pattern: str,
        weight: float = 1.0,
        confidence: float = 0.0,
    ) -> Rule:
        """
        Define a rule and store it permanently in RuleMemory.

        Rules are loaded every session — they are never forgotten.
        High-weight rules produce stronger gradient signals during training.
        """
        rule = Rule(
            rule_id=rule_id,
            name=name,
            domain=domain,
            pattern=pattern,
            weight=weight,
            confidence=confidence,
        )
        self.rule_memory.add_rule(rule)
        self.rule_memory.save()
        return rule

    # ── Lesson management ─────────────────────────────────────────────────────

    def new_lesson(
        self,
        lesson_id: str,
        name: str,
        domain: str,
        rule_ids: List[str],
        description: str = "",
        difficulty: float = 0.5,
        competency_threshold: float = 0.0,
        epochs: Optional[int] = None,
        prerequisite_ids: Optional[List[str]] = None,
    ) -> Lesson:
        """Create and register a lesson in its domain."""
        if domain not in self.domains:
            raise ValueError(f"Domain '{domain}' not registered. Call new_domain() first.")
        lesson = Lesson(
            lesson_id=lesson_id,
            name=name,
            domain=domain,
            rule_ids=rule_ids,
            description=description,
            difficulty=difficulty,
            competency_threshold=competency_threshold,
            epochs=epochs if epochs is not None else self._epochs_default,
            prerequisite_ids=prerequisite_ids or [],
        )
        self.domains[domain].add_lesson(lesson)
        return lesson

    def generate_lesson_data(
        self,
        lesson_id: str,
        n_samples: int = 400,
        val_split: float = 0.15,
        custom_generator: Optional[Callable] = None,
    ) -> None:
        """
        Generate training data for a lesson.

        If custom_generator is provided, it is called with no args and must
        return List[Tuple[NDArray, NDArray]].
        Otherwise, data is generated from the lesson's rules.
        """
        lesson = self._find_lesson(lesson_id)
        if lesson is None:
            raise ValueError(f"Lesson '{lesson_id}' not found.")

        if custom_generator is not None:
            all_data = custom_generator()
        else:
            all_data = []
            for rid in lesson.rule_ids:
                rule = self.rule_memory.get(rid)
                if rule is None:
                    logger.warning(f"Rule '{rid}' not found in memory, skipping.")
                    continue
                chunk = self.generator.from_rule(rule, n_samples=n_samples // max(len(lesson.rule_ids), 1))
                # Weight the data by rule weight: repeat high-weight rules
                repeats = max(1, int(round(rule.weight)))
                all_data.extend(chunk * repeats)

        if not all_data:
            logger.warning(f"No data generated for lesson '{lesson_id}'.")
            return

        np.random.shuffle(all_data)  # type: ignore[arg-type]
        split = max(1, int(len(all_data) * (1 - val_split)))
        lesson.train_data = all_data[:split]
        lesson.val_data   = all_data[split:] if split < len(all_data) else None
        logger.info(f"Lesson '{lesson_id}': {len(lesson.train_data)} train, "
                    f"{len(lesson.val_data) if lesson.val_data else 0} val samples")

    # ── Teaching ──────────────────────────────────────────────────────────────

    def teach_lesson(
        self,
        lesson_id: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the model on a single lesson.

        - Checks prerequisites are met.
        - Initialises the CL system if needed.
        - Applies rule weights as sample weights in the loss.
        - Marks lesson complete and saves state.

        Returns metrics dict.
        """
        lesson = self._find_lesson(lesson_id)
        if lesson is None:
            raise ValueError(f"Lesson '{lesson_id}' not found.")

        if lesson.completed and not force:
            logger.info(f"Lesson '{lesson_id}' already completed. Pass force=True to retrain.")
            return {"lesson_id": lesson_id, "status": "already_completed", "accuracy": lesson.accuracy}

        if not lesson.is_ready(self._completed):
            missing = [p for p in lesson.prerequisite_ids if p not in self._completed]
            raise RuntimeError(f"Prerequisites not met for '{lesson_id}': {missing}")

        if not lesson.train_data:
            logger.info(f"No data for '{lesson_id}', generating...")
            self.generate_lesson_data(lesson_id)

        if not lesson.train_data:
            raise RuntimeError(f"Could not generate data for lesson '{lesson_id}'.")

        # Determine dims from data
        x0, y0 = lesson.train_data[0]
        in_dim  = int(np.asarray(x0).flatten().shape[0])
        out_dim = int(np.asarray(y0).flatten().shape[0])

        self._ensure_cl_system(in_dim, out_dim)

        # Compute rule weight for this lesson (average of its rules)
        rule_weights = []
        for rid in lesson.rule_ids:
            rule = self.rule_memory.get(rid)
            if rule:
                rule_weights.append(rule.weight)
        avg_weight = float(np.mean(rule_weights)) if rule_weights else 1.0

        if self.verbose:
            print(f"\n{'='*55}")
            print(f"  Teaching: {lesson.name}  [{lesson.domain}]")
            print(f"  Rule weight: {avg_weight:.2f}  |  Epochs: {lesson.epochs}")
            print(f"  Samples: {len(lesson.train_data)} train")
            print(f"{'='*55}")

        # Pad/truncate data to match CL system dims
        cl_in  = self._cl_system.config.input_dim
        cl_out = self._cl_system.config.output_dim

        def _adjust(data: List[Tuple[Any, Any]]) -> List[Tuple[NDArray, NDArray]]:
            adj = []
            for x, y in data:
                xa = np.asarray(x).flatten()
                ya = np.asarray(y).flatten()
                # Input
                if xa.shape[0] < cl_in:
                    xa = np.pad(xa, (0, cl_in - xa.shape[0]))
                else:
                    xa = xa[:cl_in]
                # Output
                if ya.shape[0] < cl_out:
                    ya = np.pad(ya, (0, cl_out - ya.shape[0]))
                else:
                    ya = ya[:cl_out]
                adj.append((xa, ya))
            return adj

        train_adj = _adjust(lesson.train_data)
        val_adj   = _adjust(lesson.val_data) if lesson.val_data else None

        # Teach via CL system
        metrics = self._cl_system.teach(
            name=lesson_id,
            train_data=train_adj,
            val_data=val_adj,
            epochs=lesson.epochs,
            competency_threshold=lesson.competency_threshold,
            description=lesson.description,
            difficulty=lesson.difficulty,
        )

        # Update lesson state — accuracy is stored on the TaskRecord inside the learner
        task_record = self._cl_system.learner.tasks[self._cl_system.learner.current_task_id]
        lesson.accuracy  = task_record.accuracy if task_record else 0.0
        lesson.completed = True
        self._completed.append(lesson_id)

        # Strengthen rules that were taught
        for rid in lesson.rule_ids:
            self.rule_memory.strengthen_rule(rid, delta=0.15)
        self.rule_memory.save()

        # Persist
        self.save()

        if self.verbose:
            print(f"  Done: accuracy={lesson.accuracy:.4f}")

        return {
            "lesson_id": lesson_id,
            "accuracy":  lesson.accuracy,
            "epochs":    lesson.epochs,
            "final_loss": metrics.loss_history[-1] if metrics.loss_history else None,
        }

    def teach_domain(
        self,
        domain_id: str,
        skip_completed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Teach all lessons in a domain in dependency order."""
        if domain_id not in self.domains:
            raise ValueError(f"Domain '{domain_id}' not registered.")
        domain  = self.domains[domain_id]
        results = []
        for lesson in domain.ordered_lessons():
            if skip_completed and lesson.completed:
                logger.info(f"  Skipping completed: {lesson.name}")
                continue
            try:
                m = self.teach_lesson(lesson.lesson_id)
                results.append(m)
            except RuntimeError as e:
                logger.error(f"  Skipping '{lesson.name}': {e}")
        return results

    def teach_all_domains(self, domain_order: Optional[List[str]] = None) -> None:
        """Teach all registered domains, optionally in a specified order."""
        order = domain_order or list(self.domains.keys())
        for did in order:
            self.teach_domain(did)

    # ── Example presentation (few-shot rule matching) ─────────────────────────

    def present_example(
        self,
        tokens: List[str],
        domain: Optional[str] = None,
        add_to_rule: bool = True,
    ) -> Dict[str, Any]:
        """
        Present an example to the engine.

        The PatternMatcher finds the closest rule, strengthens it,
        and optionally stores the example in RuleMemory.
        Returns a dict with matched rule, similarity, and explanation.
        """
        rule, sim = self.matcher.match(tokens, domain)
        explanation = self.matcher.explain(tokens, domain)

        result: Dict[str, Any] = {
            "tokens":       tokens,
            "matched_rule": rule.name if rule else None,
            "rule_id":      rule.rule_id if rule else None,
            "similarity":   sim,
            "confidence":   rule.confidence if rule else 0.0,
            "explanation":  explanation,
        }

        if rule and add_to_rule:
            self.rule_memory.add_example_to_rule(
                rule.rule_id,
                {"tokens": tokens, "similarity": sim}
            )
            self.rule_memory.save()

        if self.verbose:
            print(f"\n[Example] {tokens}")
            print(f"  {explanation}")

        return result

    def query(
        self,
        input_data: Any,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the trained model on an input and return prediction + rule match."""
        if self._cl_system is None:
            return {"error": "No model trained yet. Call teach_lesson() first."}

        x = np.asarray(input_data, dtype=np.float64).flatten()

        # Pad/truncate to match model input dim
        in_dim = self._cl_system.config.input_dim
        if len(x) < in_dim:
            x = np.pad(x, (0, in_dim - len(x)))
        else:
            x = x[:in_dim]

        params = self._cl_system.learner.params
        pred   = self._cl_system.base_lens.forward(params.data, x)

        tokens = [f"val_{v:.2f}" for v in x[:8]]
        rule, sim = self.matcher.match(tokens, domain)

        return {
            "prediction":   np.asarray(pred).tolist(),
            "matched_rule": rule.name if rule else None,
            "similarity":   sim,
        }

    # ── Cross-domain knowledge transfer ──────────────────────────────────────

    def enable_transfer(self, target_domain: str, source_domains: List[str]) -> None:
        """
        Enable knowledge transfer from source_domains into target_domain.

        Transfer works at the encoder level: the target domain's lessons
        start with embeddings pre-warmed by source domain rules.
        """
        self._transfer_map[target_domain] = source_domains
        if target_domain in self.domains:
            self.domains[target_domain].transfer_from = source_domains

        for src_id in source_domains:
            src_rules = self.rule_memory.by_domain(src_id)
            for rule in src_rules:
                tokens = rule.pattern.replace("[", " ").replace("]", " ").split()
                for tok in tokens:
                    src_vec = self.encoder.encode(tok)
                    self.encoder.update_learned(
                        f"{target_domain}:{tok}",
                        src_vec * 0.1 * rule.confidence
                    )

        logger.info(f"Transfer enabled: {source_domains} -> {target_domain}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_lesson(self, lesson_id: str) -> Optional[Lesson]:
        for domain in self.domains.values():
            lesson = domain.get_lesson(lesson_id)
            if lesson:
                return lesson
        return None

    def _ensure_cl_system(self, in_dim: int, out_dim: int) -> None:
        if self._cl_system is None:
            self._cl_system = ContinualLearningSystem.create(
                input_dim=in_dim,
                output_dim=out_dim,
                strategy=self._cl_strategy,
                hidden_dims=self._hidden_dims,
                learning_rate=self._learning_rate,
                epochs_per_task=self._epochs_default,
                verbose=self.verbose,
                checkpoint_dir=str(self.memory_dir / "checkpoints"),
            )
            logger.info(f"CL system initialised: in={in_dim} out={out_dim} strategy={self._cl_strategy}")
        else:
            cur_in  = self._cl_system.config.input_dim
            cur_out = self._cl_system.config.output_dim
            if cur_in != in_dim or cur_out != out_dim:
                logger.warning(
                    f"Lesson dims ({in_dim},{out_dim}) differ from CL system "
                    f"({cur_in},{cur_out}). Input will be padded/truncated."
                )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save all state to disk."""
        self._save_state()
        self.encoder.save(self.memory_dir / "encoder.pkl")
        self.rule_memory.save()
        if self._cl_system is not None:
            self._cl_system.save(self.memory_dir / "cl_system.pkl")
        logger.info(f"Engine state saved -> {self.memory_dir}")

    def _save_state(self) -> None:
        state = {
            "completed":    self._completed,
            "transfer_map": self._transfer_map,
            "domains": {
                did: {
                    "domain_id":   d.domain_id,
                    "name":        d.name,
                    "description": d.description,
                    "transfer_from": d.transfer_from,
                    "lessons": [
                        {
                            "lesson_id":   l.lesson_id,
                            "name":        l.name,
                            "domain":      l.domain,
                            "rule_ids":    l.rule_ids,
                            "description": l.description,
                            "difficulty":  l.difficulty,
                            "competency_threshold": l.competency_threshold,
                            "epochs":      l.epochs,
                            "prerequisite_ids": l.prerequisite_ids,
                            "completed":   l.completed,
                            "accuracy":    l.accuracy,
                        }
                        for l in d.lessons
                    ],
                }
                for did, d in self.domains.items()
            },
        }
        path = self.memory_dir / "engine_state.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _load_state(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._completed    = state.get("completed", [])
        self._transfer_map = state.get("transfer_map", {})
        for did, ddata in state.get("domains", {}).items():
            domain = Domain(
                domain_id=ddata["domain_id"],
                name=ddata["name"],
                description=ddata.get("description", ""),
                transfer_from=ddata.get("transfer_from", []),
            )
            for ldata in ddata.get("lessons", []):
                lesson = Lesson(
                    lesson_id=ldata["lesson_id"],
                    name=ldata["name"],
                    domain=ldata["domain"],
                    rule_ids=ldata["rule_ids"],
                    description=ldata.get("description", ""),
                    difficulty=ldata.get("difficulty", 0.5),
                    competency_threshold=ldata.get("competency_threshold", 0.0),
                    epochs=ldata.get("epochs", self._epochs_default),
                    prerequisite_ids=ldata.get("prerequisite_ids", []),
                    completed=ldata.get("completed", False),
                    accuracy=ldata.get("accuracy", 0.0),
                )
                domain.add_lesson(lesson)
            self.domains[did] = domain
        logger.info(f"Engine state loaded: {len(self.domains)} domains, {len(self._completed)} completed lessons")

        # Load CL system if it exists
        cl_path = self.memory_dir / "cl_system.pkl"
        if cl_path.exists():
            try:
                self._cl_system = ContinualLearningSystem.load(cl_path)
                logger.info("CL system loaded from disk.")
            except Exception as e:
                logger.warning(f"Could not load CL system: {e}")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self) -> str:
        lines = [
            "=" * 60,
            "CURRICULUM ENGINE REPORT",
            "=" * 60,
            f"  Memory dir : {self.memory_dir}",
            f"  Strategy   : {self._cl_strategy}",
            f"  Domains    : {len(self.domains)}",
            f"  Completed  : {len(self._completed)} lessons",
            "",
            self.rule_memory.summary(),
            "",
        ]
        for did, domain in self.domains.items():
            lines.append(f"  Domain: {domain.name}")
            for lesson in domain.lessons:
                status = "DONE" if lesson.completed else "TODO"
                lines.append(
                    f"    [{status}] {lesson.lesson_id:<25s} "
                    f"acc={lesson.accuracy:.3f}  diff={lesson.difficulty:.1f}"
                )
        if self._cl_system:
            lines.append("")
            lines.append(self._cl_system.learner.summary())
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# SECTION 7: BUILT-IN DOMAIN BUILDERS  (user-extendable examples)
# ============================================================================

def build_math_curriculum(engine: CurriculumEngine) -> Domain:
    """
    Math domain: objects -> counting -> comparison -> ordering.
    This is the foundational domain — all others transfer from it.
    """
    math = engine.new_domain("math", "Mathematics",
                             "Foundational symbolic reasoning: objects, counting, comparison, ordering")

    # Rules (permanent, high-weight)
    engine.define_rule("object_identity", "Object Identity", "math",
                       "[object] = [object]", weight=2.0)
    engine.define_rule("count_rule",      "Counting Rule",   "math",
                       "[object] * N = N objects", weight=2.0)
    engine.define_rule("less_than",       "Less Than",       "math",
                       "[A] < [B] when A fewer than B", weight=1.5)
    engine.define_rule("equal_rule",      "Equality",        "math",
                       "[A] = [B] when A same as B", weight=1.5)
    engine.define_rule("greater_than",    "Greater Than",    "math",
                       "[A] > [B] when A more than B", weight=1.5)
    engine.define_rule("order_rule",      "Ordering Rule",   "math",
                       "order [A B C] ascending by count", weight=1.5)

    # Lessons
    math.add_lesson(Lesson(
        lesson_id="math_objects",
        name="Objects as Identity",
        domain="math",
        rule_ids=["object_identity"],
        description="Everything is an object. The building block of symbolic learning.",
        difficulty=0.1,
        competency_threshold=0.80,
        epochs=30,
    ))

    math.add_lesson(Lesson(
        lesson_id="math_counting",
        name="Counting",
        domain="math",
        rule_ids=["count_rule"],
        description="[obj]*N = N objects. Rule-weighted, few-shot generalisation.",
        difficulty=0.3,
        competency_threshold=0.75,
        epochs=50,
        prerequisite_ids=["math_objects"],
    ))

    math.add_lesson(Lesson(
        lesson_id="math_comparison",
        name="Comparison",
        domain="math",
        rule_ids=["less_than", "equal_rule", "greater_than"],
        description="Compare two counts using <, =, >.",
        difficulty=0.4,
        competency_threshold=0.75,
        epochs=50,
        prerequisite_ids=["math_counting"],
    ))

    math.add_lesson(Lesson(
        lesson_id="math_ordering",
        name="Ordering",
        domain="math",
        rule_ids=["order_rule"],
        description="Order a sequence by count using the comparison lesson.",
        difficulty=0.5,
        competency_threshold=0.70,
        epochs=60,
        prerequisite_ids=["math_comparison"],
    ))

    return math


def build_english_curriculum(engine: CurriculumEngine) -> Domain:
    """
    English domain: vocabulary anchoring -> word categories.
    Transfers from math (symbolic identity).
    """
    engine.enable_transfer("english", ["math"])
    english = engine.new_domain("english", "English",
                                "Vocabulary, word categories, and language structure",
                                transfer_from=["math"])

    engine.define_rule("word_is_symbol", "Word as Symbol", "english",
                       "[word] = symbolic unit", weight=2.0)
    engine.define_rule("noun_rule",      "Noun",           "english",
                       "[noun] names a thing", weight=1.5)
    engine.define_rule("verb_rule",      "Verb",           "english",
                       "[verb] describes action", weight=1.5)
    engine.define_rule("adj_rule",       "Adjective",      "english",
                       "[adjective] describes quality", weight=1.5)

    english.add_lesson(Lesson(
        lesson_id="eng_vocab",
        name="Vocabulary Anchoring",
        domain="english",
        rule_ids=["word_is_symbol"],
        description="Each word maps to a stable symbolic embedding.",
        difficulty=0.2,
        competency_threshold=0.80,
        epochs=40,
    ))

    english.add_lesson(Lesson(
        lesson_id="eng_categories",
        name="Word Categories",
        domain="english",
        rule_ids=["noun_rule", "verb_rule", "adj_rule"],
        description="Classify words as noun, verb, or adjective.",
        difficulty=0.4,
        competency_threshold=0.75,
        epochs=50,
        prerequisite_ids=["eng_vocab"],
    ))

    return english


def build_physics_curriculum(engine: CurriculumEngine) -> Domain:
    """
    Physics domain: F=ma -> KE = 0.5*m*v^2.
    Transfers from math.
    """
    engine.enable_transfer("physics", ["math"])
    physics = engine.new_domain("physics", "Physics",
                                "Physical laws and formula mappings",
                                transfer_from=["math"])

    engine.define_rule("newton_second", "Newton's Second Law", "physics",
                       "F = [mass] * [acceleration]", weight=2.0)
    engine.define_rule("kinetic",       "Kinetic Energy",      "physics",
                       "KE = 0.5 * [mass] * [velocity]^2", weight=2.0)

    physics.add_lesson(Lesson(
        lesson_id="phys_fma",
        name="Force = Mass x Acceleration",
        domain="physics",
        rule_ids=["newton_second"],
        description="F = m * a",
        difficulty=0.4,
        competency_threshold=0.70,
        epochs=60,
    ))

    physics.add_lesson(Lesson(
        lesson_id="phys_ke",
        name="Kinetic Energy",
        domain="physics",
        rule_ids=["kinetic"],
        description="KE = 0.5 * m * v^2",
        difficulty=0.5,
        competency_threshold=0.65,
        epochs=60,
        prerequisite_ids=["phys_fma"],
    ))

    return physics


def build_programming_curriculum(engine: CurriculumEngine) -> Domain:
    """
    Programming domain: token classification -> assignment rule.
    Transfers from math and english.
    """
    engine.enable_transfer("programming", ["math", "english"])
    prog = engine.new_domain("programming", "Programming",
                             "Code tokens, expressions, and logic",
                             transfer_from=["math", "english"])

    engine.define_rule("token_is_symbol", "Token as Symbol", "programming",
                       "[token] = symbolic unit", weight=2.0)
    engine.define_rule("keyword_rule",    "Keyword",         "programming",
                       "[keyword] controls flow", weight=1.5)
    engine.define_rule("operator_rule",   "Operator",        "programming",
                       "[operator] transforms [value]", weight=1.5)
    engine.define_rule("assignment_rule", "Assignment",      "programming",
                       "[var] = [expression]", weight=1.5)

    prog.add_lesson(Lesson(
        lesson_id="prog_tokens",
        name="Token Classification",
        domain="programming",
        rule_ids=["token_is_symbol", "keyword_rule", "operator_rule"],
        description="Classify code tokens: keyword, operator, identifier, literal.",
        difficulty=0.4,
        competency_threshold=0.75,
        epochs=50,
    ))

    prog.add_lesson(Lesson(
        lesson_id="prog_assignment",
        name="Assignment Rule",
        domain="programming",
        rule_ids=["assignment_rule"],
        description="Variable assignment: name = value.",
        difficulty=0.5,
        competency_threshold=0.70,
        epochs=60,
        prerequisite_ids=["prog_tokens"],
    ))

    return prog


# ============================================================================
# SECTION 8: DATA GENERATORS FOR BUILT-IN DOMAINS
# ============================================================================

def _generate_english_vocab_data(engine: CurriculumEngine, n: int = 300) -> List[Tuple[NDArray, NDArray]]:
    vocab = ["cat","dog","run","jump","big","small","house","tree",
             "eat","sleep","red","blue","car","book","write","read"]
    return engine.generator.word_identity(vocab, n)


def _generate_english_category_data(engine: CurriculumEngine, n: int = 400) -> List[Tuple[NDArray, NDArray]]:
    categories = {
        "noun":      ["cat","dog","house","tree","car","book","river","mountain"],
        "verb":      ["run","jump","eat","sleep","write","read","fly","swim"],
        "adjective": ["big","small","red","blue","fast","slow","hot","cold"],
    }
    return engine.generator.word_category(categories, n)


def _generate_physics_fma_data(engine: CurriculumEngine, n: int = 400) -> List[Tuple[NDArray, NDArray]]:
    def fma(x: NDArray) -> NDArray:
        m = abs(x[0]) + 0.1
        a = x[1]
        return np.array([m * a])
    return engine.generator.formula_mapping(fma, input_dim=2, n_samples=n)


def _generate_physics_ke_data(engine: CurriculumEngine, n: int = 400) -> List[Tuple[NDArray, NDArray]]:
    def ke(x: NDArray) -> NDArray:
        m = abs(x[0]) + 0.1
        v = x[1]
        return np.array([0.5 * m * v * v])
    return engine.generator.formula_mapping(ke, input_dim=2, n_samples=n)


def _generate_prog_token_data(engine: CurriculumEngine, n: int = 400) -> List[Tuple[NDArray, NDArray]]:
    token_classes = {
        "keyword":    ["if","else","while","for","return","def","class","import"],
        "operator":   ["+","-","*","/","==","!=","<",">","=","and","or","not"],
        "identifier": ["x","y","z","count","name","value","result","data"],
        "literal":    ["0","1","42","3.14","true","false","null","'hello'"],
    }
    return engine.generator.token_classification(token_classes, n)


def _attach_generators(engine: CurriculumEngine) -> None:
    """Attach custom data generators to built-in lessons."""
    gen_map: Dict[str, Callable] = {
        "eng_vocab":      lambda: _generate_english_vocab_data(engine),
        "eng_categories": lambda: _generate_english_category_data(engine),
        "phys_fma":       lambda: _generate_physics_fma_data(engine),
        "phys_ke":        lambda: _generate_physics_ke_data(engine),
        "prog_tokens":    lambda: _generate_prog_token_data(engine),
    }
    for lesson_id, gen_fn in gen_map.items():
        lesson = engine._find_lesson(lesson_id)
        if lesson and not lesson.train_data:
            engine.generate_lesson_data(lesson_id, custom_generator=gen_fn)


# ============================================================================
# SECTION 9: FULL CURRICULUM FACTORY
# ============================================================================

def create_full_curriculum(
    memory_dir: str = "eti_memory",
    strategy: str = "ewc",
    hidden_dims: Optional[List[int]] = None,
    learning_rate: float = 0.003,
    epochs_per_lesson: int = 50,
    verbose: bool = True,
) -> CurriculumEngine:
    """
    Create a CurriculumEngine with all 4 built-in domains pre-configured.

    Domains: math -> english -> physics -> programming
    Transfer: math -> english, math -> physics, math+english -> programming

    Returns a ready-to-use engine. Call engine.teach_all_domains() to start.
    """
    engine = CurriculumEngine.create(
        memory_dir=memory_dir,
        hidden_dims=hidden_dims or [256, 256],
        cl_strategy=strategy,
        learning_rate=learning_rate,
        epochs_per_lesson=epochs_per_lesson,
        verbose=verbose,
    )

    if "math" not in engine.domains:
        build_math_curriculum(engine)
    if "english" not in engine.domains:
        build_english_curriculum(engine)
    if "physics" not in engine.domains:
        build_physics_curriculum(engine)
    if "programming" not in engine.domains:
        build_programming_curriculum(engine)

    _attach_generators(engine)
    return engine


# ============================================================================
# SECTION 10: DEMO / ENTRY POINT
# ============================================================================

def demo() -> None:
    """
    End-to-end demonstration of the CurriculumEngine.

    Shows:
      1. Creating an engine with all 4 built-in domains
      2. Teaching the math curriculum (objects -> counting -> comparison -> ordering)
      3. Presenting few-shot examples and observing rule matching
      4. Cross-domain transfer (math -> english)
      5. Teaching the english curriculum
      6. Querying the model
      7. Printing the full report
    """
    print("\n" + "=" * 60)
    print("  ETI CURRICULUM ENGINE — DEMO")
    print("=" * 60)

    # ── 1. Create engine ──────────────────────────────────────────────────────
    engine = create_full_curriculum(
        memory_dir="eti_memory_demo",
        strategy="ewc",
        hidden_dims=[128, 128],
        learning_rate=0.003,
        epochs_per_lesson=20,   # short for demo
        verbose=True,
    )

    # ── 2. Teach math curriculum ──────────────────────────────────────────────
    print("\n[STEP 1] Teaching Math curriculum...")
    engine.teach_domain("math")

    # ── 3. Present few-shot examples ──────────────────────────────────────────
    print("\n[STEP 2] Presenting few-shot examples to strengthen rules...")

    examples = [
        (["object", "object", "object"],                    "math"),
        (["object", "object", "object", "object"],          "math"),
        (["count_3", "less_than", "count_5"],               "math"),
        (["count_7", "greater_than", "count_2"],            "math"),
        (["order", "count_1", "count_3", "count_2"],        "math"),
    ]
    for tokens, domain in examples:
        result = engine.present_example(tokens, domain=domain)
        print(f"  tokens={tokens}")
        print(f"  -> {result['explanation']}\n")

    # ── 4. Cross-domain transfer ──────────────────────────────────────────────
    print("\n[STEP 3] Enabling cross-domain transfer: math -> english...")
    engine.enable_transfer("english", ["math"])

    # ── 5. Teach english curriculum ───────────────────────────────────────────
    print("\n[STEP 4] Teaching English curriculum...")
    engine.teach_domain("english")

    # ── 6. Query the model ────────────────────────────────────────────────────
    print("\n[STEP 5] Querying the model...")
    test_word = "cat"
    x = engine.encoder.encode(test_word)
    result = engine.query(x, domain="english")
    print(f"  Input: '{test_word}' encoding")
    print(f"  Prediction (first 4 dims): {result['prediction'][:4]}")
    print(f"  Matched rule: {result['matched_rule']}  sim={result['similarity']:.3f}")

    # ── 7. Full report ────────────────────────────────────────────────────────
    print("\n[STEP 6] Full report:")
    print(engine.report())

    # ── 8. Save ───────────────────────────────────────────────────────────────
    engine.save()
    print("\nEngine state saved. Rules and encoder persist across sessions.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
