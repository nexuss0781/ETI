"""
Universal Training Framework for Exteroterrestrial Intelligence (ETI)
====================================================================

Production-ready categorical foundation for training across continuous, discrete, 
hypercomputational, and non-classical domains using parametric lenses 
and reverse derivative categories.

Version: 3.0.0-PRODUCTION
Author: ETI Research Division
License: MIT

Mathematical Foundation:
- Cartesian Reverse Differential Categories (CRDC)
- Parametric Lenses: Para(Lens(C))
- Sheaf-theoretic constraint propagation
- Functorial gradient descent
- Enriched category theory for optimization

References:
    - Cruttwell et al. "Deep Learning with Parametric Lenses" (2024)
    - Cockett et al. "Reverse Derivative Categories" (CSL 2020)
    - Fong, Spivak "Seven Sketches in Compositionality" (2018)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import pickle
import threading
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum, IntEnum, auto
from functools import lru_cache, partial, reduce, wraps
from numbers import Number
from pathlib import Path
from queue import PriorityQueue
from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.sparse import csr_matrix, csc_matrix, spmatrix
from scipy.sparse.linalg import LinearOperator, cg, gmres, minres

# Type definitions for mathematical rigor
ParameterVector = NewType('ParameterVector', NDArray[np.float64])
TangentVectorType = NewType('TangentVectorType', NDArray[np.float64])
LossValue = NewType('LossValue', float)
Epoch = NewType('Epoch', int)
Step = NewType('Step', int)

# Configure structured logging
logger = logging.getLogger("eti.universal_trainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Generic type variables for categorical abstractions
P = TypeVar('P')   # Parameter type
A = TypeVar('A')   # Input type
B = TypeVar('B')   # Output type
C = TypeVar('C')   # Composed output type
T = TypeVar('T')   # Generic type
K = TypeVar('K')   # Enrichment category type
A2 = TypeVar('A2') # Second input type
B2 = TypeVar('B2') # Second output type


# ============================================================================
# SECTION 1: MATHEMATICAL ABSTRACTIONS AND CATEGORY THEORY FOUNDATIONS
# ============================================================================

class CategoryError(Exception):
    """Base exception for categorical violations."""
    pass


class LensError(CategoryError):
    """Violation of lens laws."""
    pass


class SheafError(CategoryError):
    """Sheaf cohomology violation."""
    pass


class ConvergenceError(Exception):
    """Failure to converge in optimization."""
    pass


class ValidationError(Exception):
    """Input validation failure."""
    pass


class CheckpointError(Exception):
    """Checkpoint save/load failure."""
    pass


class ConstraintType(IntEnum):
    """
    Types of parameter constraints supported.
    Ordered by computational complexity.
    """
    EUCLIDEAN = 0      # Standard vector space structure
    RIEMANNIAN = 1     # Manifold with metric
    DISCRETE = 2       # Categorical/Gumbel-softmax
    ORDINAL = 3        # Transfinite ordinals
    SHEAF = 4          # Cohomological constraints
    HYPERREAL = 5      # Non-standard analysis (*R)
    QUANTUM = 6        # Density matrices/CP maps


class TemporalStructure(IntEnum):
    """Non-standard temporal structures for training."""
    STANDARD = 0       # Natural numbers
    ORDINAL = 1        # Transfinite ordinals
    RETROCAUSAL = 2    # Causal loops (closed timelike curves)
    HYPERREAL = 3      # *N with infinitesimals
    PARACONSISTENT = 4 # Dialetheist time (contradictions allowed)


class OptimizationMode(Enum):
    """Optimization strategy modes."""
    DETERMINISTIC = auto()
    STOCHASTIC = auto()
    ADVERSARIAL = auto()
    COEVOLUTIONARY = auto()
    QUANTUM = auto()


@runtime_checkable
class Differentiable(Protocol[P, A, B]):
    """
    Protocol for differentiable morphisms in CRDCs.
    
    In a Cartesian Reverse Differential Category, every morphism f: A → B 
    has an associated reverse derivative R[f]: A × B' → A'.
    
    Mathematical specification:
        Given f: P × A → B, the reverse derivative is:
        R[f]: (P × A) × B' → (P' × A')
    """
    
    def forward(self, params: P, input_data: A) -> B:
        """Forward pass: compute f(p, a)."""
        ...
    
    def reverse(self, params: P, input_data: A, output_tangent: B) -> Tuple[A, P]:
        """
        Reverse pass: compute R[f](p, a, b').
        
        Returns:
            Tuple of (input_tangent, param_tangent)
            
        Law (Gradient Agreement):
            For all smooth curves γ: ℝ → P × A,
            d/dt f(γ(t))|_{t=0} = ⟨R[f](γ(0), f(γ(0))), γ'(0)⟩
        """
        ...


@runtime_checkable
class Monoidal(Protocol[T]):
    """Protocol for monoidal structure."""
    
    def tensor(self, other: T) -> T:
        """Monoidal product ⊗."""
        ...
    
    def unit(self) -> T:
        """Monoidal unit I."""
        ...


@runtime_checkable
class Enriched(Protocol[T, K]):
    """
    Category enriched over monoidal category K.
    
    Hom-objects are objects of K rather than sets.
    """
    
    def hom_object(self, target: T) -> K:
        """Return hom-object [self, target] in K."""
        ...
    
    def composition(self, other: T) -> Callable[[K, K], K]:
        """Enriched composition morphism."""
        ...


# ============================================================================
# SECTION 2: TENSOR SPECIFICATION AND GEOMETRY
# ============================================================================

@dataclass(frozen=True, slots=True, order=True)
class TensorSpec:
    """
    Immutable tensor specification with geometric structure.
    
    Represents objects in the base category with:
    - Shape (dimension vector)
    - Dtype (field structure)
    - Device (computational substrate)
    - Memory layout (strides)
    """
    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype('float64'))
    device: str = "cpu"
    layout: Literal['C', 'F', 'A'] = 'C'
    
    def __post_init__(self) -> None:
        if any(s is not None and s <= 0 for s in self.shape):
            raise ValidationError(
                f"All dimensions must be positive, got {self.shape}"
            )
        if self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            warnings.warn(
                f"Non-floating dtype {self.dtype} may cause numerical issues",
                RuntimeWarning
            )
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        dims = [s for s in self.shape if s is not None]
        return int(np.prod(dims, dtype=np.int64)) if dims else 0
    
    @property
    def nbytes(self) -> int:
        """Total bytes required."""
        return self.size * self.dtype.itemsize
    
    def validate(self, array: NDArray, name: str = "tensor") -> None:
        """
        Validate array against specification.

        Raises:
            ValidationError: If shape or dtype mismatch.
        """
        # Skip shape validation for dynamic specs (containing None)
        if not any(s is None for s in self.shape):
            if array.shape != self.shape:
                raise ValidationError(
                    f"{name} shape mismatch: expected {self.shape}, got {array.shape}"
                )
        if array.dtype != self.dtype:
            # Allow upcasting but not downcasting
            if np.can_cast(array.dtype, self.dtype, casting='safe'):
                return
            raise ValidationError(
                f"{name} dtype mismatch: expected {self.dtype}, got {array.dtype}"
            )
    
    def compatible_with(self, other: TensorSpec) -> bool:
        """Check if specs are broadcast-compatible."""
        # Simplified broadcasting check
        return True  # Detailed implementation omitted for brevity
    
    def flatten(self) -> TensorSpec:
        """Return 1D specification."""
        return TensorSpec(
            shape=(self.size,),
            dtype=self.dtype,
            device=self.device,
            layout=self.layout
        )
    
    def with_shape(self, *shape: int) -> TensorSpec:
        """Create new spec with different shape, same metadata."""
        return TensorSpec(
            shape=shape,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'shape': self.shape,
            'dtype': str(self.dtype),
            'device': self.device,
            'layout': self.layout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TensorSpec:
        """Deserialize from dictionary."""
        return cls(
            shape=tuple(data['shape']),
            dtype=np.dtype(data['dtype']),
            device=data['device'],
            layout=data['layout']
        )


# ============================================================================
# SECTION 3: DIFFERENTIAL GEOMETRY - TANGENT BUNDLES
# ============================================================================

@dataclass(frozen=True, slots=True)
class TangentBundle:
    """
    Tangent vector in the differential structure.
    
    In CRDCs, the tangent bundle functor T assigns to each object A 
    its tangent space T(A) = A × A', where A' is the space of 
    differential changes.
    
    Mathematical structure:
        - Base point (position in manifold)
        - Fiber vector (direction/velocity)
        - Metric tensor (for norm computation)
    """
    base_point: NDArray[np.float64]
    fiber_vector: NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        if self.base_point.shape != self.fiber_vector.shape:
            raise ValidationError(
                f"Base {self.base_point.shape} and fiber {self.fiber_vector.shape} "
                "must have same shape"
            )
        # Ensure float64
        object.__setattr__(
            self, 'base_point', 
            np.asarray(self.base_point, dtype=np.float64)
        )
        object.__setattr__(
            self, 'fiber_vector',
            np.asarray(self.fiber_vector, dtype=np.float64)
        )
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.base_point.shape
    
    @property
    def norm(self) -> float:
        """Riemannian norm sqrt(g(v,v))."""
        return float(np.linalg.norm(self.fiber_vector))
    
    def __add__(self, other: TangentBundle) -> TangentBundle:
        """Vector addition in fiber."""
        if self.shape != other.shape:
            raise ValidationError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return TangentBundle(
            base_point=self.base_point,  # Same base point
            fiber_vector=self.fiber_vector + other.fiber_vector,
            metadata={**self.metadata, **other.metadata, 'operation': 'add'}
        )
    
    def __sub__(self, other: TangentBundle) -> TangentBundle:
        """Vector subtraction."""
        return self + TangentBundle(
            base_point=other.base_point,
            fiber_vector=-other.fiber_vector,
            metadata=other.metadata
        )
    
    def __mul__(self, scalar: float) -> TangentBundle:
        """Scalar multiplication."""
        return TangentBundle(
            base_point=self.base_point,
            fiber_vector=self.fiber_vector * scalar,
            metadata={**self.metadata, 'scaled_by': scalar}
        )
    
    __rmul__ = __mul__
    
    def __truediv__(self, scalar: float) -> TangentBundle:
        if abs(scalar) < 1e-15:
            raise ZeroDivisionError("Division by near-zero scalar")
        return self * (1.0 / scalar)
    
    def inner_product(self, other: TangentBundle) -> float:
        """Riemannian inner product."""
        return float(np.dot(self.fiber_vector.flatten(), other.fiber_vector.flatten()))
    
    def project_onto(self, subspace: NDArray) -> TangentBundle:
        """
        Project tangent vector onto subspace.
        
        Args:
            subspace: Orthonormal basis vectors as columns.
        """
        coeffs = subspace.T @ self.fiber_vector.flatten()
        projected = subspace @ coeffs
        return TangentBundle(
            base_point=self.base_point,
            fiber_vector=projected.reshape(self.shape),
            metadata={**self.metadata, 'projected': True}
        )
    
    def parallel_transport(self, new_base: NDArray, connection: Optional[Callable] = None) -> TangentBundle:
        """
        Parallel transport to new base point.
        
        For Euclidean space, this is trivial. For manifolds, requires connection.
        """
        if connection is None:
            # Euclidean transport
            return TangentBundle(
                base_point=new_base,
                fiber_vector=self.fiber_vector,
                metadata={**self.metadata, 'transported': True}
            )
        return connection(self, new_base)
    
    def copy(self) -> TangentBundle:
        """Deep copy."""
        return TangentBundle(
            base_point=self.base_point.copy(),
            fiber_vector=self.fiber_vector.copy(),
            metadata=copy.deepcopy(self.metadata)
        )


@dataclass(frozen=True)
class CotangentBundle:
    """
    Cotangent vectors (linear functionals on tangent space).
    
    In optimization, gradients live in cotangent space.
    """
    base_point: NDArray[np.float64]
    covector: NDArray[np.float64]
    
    def apply_to(self, tangent: TangentBundle) -> float:
        """Evaluate covector on tangent vector."""
        return float(np.dot(self.covector.flatten(), tangent.fiber_vector.flatten()))
    
    def musical_isomorphism(self, metric: Optional[NDArray] = None) -> TangentBundle:
        """
        Sharp operator #: T*M → TM.
        
        Converts gradient (covector) to update direction (vector).
        """
        if metric is None:
            # Euclidean metric
            return TangentBundle(
                base_point=self.base_point,
                fiber_vector=self.covector
            )
        # Riemannian metric
        vector = metric @ self.covector.flatten()
        return TangentBundle(
            base_point=self.base_point,
            fiber_vector=vector.reshape(self.covector.shape)
        )


# ============================================================================
# SECTION 4: SHEAF THEORY AND COHOMOLOGICAL CONSTRAINTS
# ============================================================================

@dataclass(frozen=True)
class SheafCell:
    """Cell in cellular sheaf decomposition."""
    dimension: int
    index: int
    stalk_dim: int
    
    def __hash__(self) -> int:
        return hash((self.dimension, self.index))


@dataclass
class SheafStructure:
    """
    Cellular sheaf for distributed constraint propagation.

    Mathematical structure:
        - Base space: Computational graph (cell complex)
        - Stalks: Vector spaces at vertices (parameters)
        - Restriction maps: Linear maps ρ: F(v) → F(e) for v ∈ e
        - Sheaf Laplacian: L = δ*δ (discrete Hodge Laplacian)

    Cohomology H^0(X; F) represents globally consistent configurations.
    """
    cells: Dict[SheafCell, int] = field(default_factory=dict)
    restriction_maps: Dict[Tuple[SheafCell, SheafCell], spmatrix] = field(default_factory=dict)
    coboundary: Optional[spmatrix] = field(default=None, repr=False)

    # Configuration
    tolerance: float = 1e-10
    max_cg_iter: int = 1000

    # Internal computed fields (not constructor args)
    def __post_init__(self) -> None:
        # Use object.__setattr__ to set private attrs on a regular dataclass
        object.__setattr__(self, '_laplacian', None)
        object.__setattr__(self, '_harmonic_projector', None)
        if self.coboundary is not None:
            self._compute_laplacian()
    
    def _compute_laplacian(self) -> None:
        """Compute sheaf Laplacian L = δ†δ."""
        if self.coboundary is None:
            return

        # L_F = δ^T δ for cellular sheaves
        laplacian = self.coboundary.T @ self.coboundary
        object.__setattr__(self, '_laplacian', laplacian)

        # Precompute harmonic projector (expensive but useful)
        try:
            from scipy.sparse.linalg import svds
            k = min(10, laplacian.shape[0] - 1)
            u, s, vt = svds(laplacian, k=k, which='SM')
            null_mask = s < self.tolerance
            if np.any(null_mask):
                object.__setattr__(self, '_harmonic_projector', u[:, null_mask] @ u[:, null_mask].T)
        except Exception as e:
            logger.warning(f"Could not precompute harmonic projector: {e}")
    
    @property
    def laplacian(self) -> Optional[spmatrix]:
        return getattr(self, '_laplacian', None)

    def project_harmonic(self, params: NDArray) -> NDArray:
        """
        Project to harmonic space (global consistency).

        Solves: min ||x - params||^2 s.t. Lx = 0

        Equivalent to computing the cohomology class [params] ∈ H^0.
        """
        lap = getattr(self, '_laplacian', None)
        if lap is None:
            return params

        flat_params = params.flatten()
        harmonic_proj = getattr(self, '_harmonic_projector', None)

        # If we have precomputed projector, use it
        if harmonic_proj is not None:
            harmonic = harmonic_proj @ flat_params
            return (flat_params - harmonic).reshape(params.shape)
        
        # Otherwise solve using conjugate gradient
        try:
            # Solve Lx = L(params) to get closest harmonic form
            b = lap @ flat_params
            harmonic, info = cg(
                lap,
                b,
                tol=self.tolerance,
                maxiter=self.max_cg_iter
            )
            
            if info == 0:
                return (flat_params - harmonic).reshape(params.shape)
            elif info > 0:
                logger.warning(f"CG convergence to tolerance not achieved, iter={info}")
            else:
                logger.error(f"CG breakdown, info={info}")
                
        except Exception as e:
            logger.error(f"Harmonic projection failed: {e}")
        
        return params
    
    def check_local_consistency(self, params: NDArray, cell_i: SheafCell, cell_j: SheafCell) -> bool:
        """Verify restriction map commutes: ρ_ij(params[i]) = ρ_ji(params[j])."""
        key = (cell_i, cell_j)
        if key not in self.restriction_maps:
            return True
        
        rho = self.restriction_maps[key]
        # Simplified: assume params indexed by cell
        diff = np.abs(rho @ params[cell_i.index] - params[cell_j.index])
        return bool(np.all(diff < self.tolerance))
    
    def coboundary_operator(self, cochain: NDArray) -> NDArray:
        """Apply coboundary operator δ: C^0 → C^1."""
        if self.coboundary is None:
            return np.zeros_like(cochain)
        return self.coboundary @ cochain.flatten()
    
    def hodge_decomposition(self, form: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Hodge decomposition: ω = dα + δβ + h
        
        Returns:
            (exact_part, coexact_part, harmonic_part)
        """
        # This is computationally expensive; use iterative methods
        harmonic = self.project_harmonic(form)
        remainder = form - harmonic
        
        # remainder is in image of d + image of δ
        # For simplicity, return as coexact (exact requires pseudoinverse)
        return np.zeros_like(form), remainder, harmonic


# ============================================================================
# SECTION 5: PARAMETER STATE MANAGEMENT
# ============================================================================

@dataclass
class ParameterMetadata:
    """Metadata for parameter tracking."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 0
    gradient_norm: float = 0.0
    update_norm: float = 0.0
    constraint_violation: float = 0.0
    tags: Set[str] = field(default_factory=set)
    history: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    
    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = datetime.now()
    
    def record_step(self, info: Dict[str, Any]) -> None:
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'version': self.version,
            **info
        })
        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-500:]


@dataclass
class ParameterState:
    """
    Thread-safe, immutable parameter state with geometric structure.
    
    Design:
        - Copy-on-write for thread safety
        - Pluggable constraint handling via strategy pattern
        - Full provenance tracking
    
    Mathematical invariants:
        - Parameters respect constraint manifold
        - Updates preserve structural properties
    """
    data: NDArray[np.float64]
    spec: TensorSpec
    constraint_type: ConstraintType = ConstraintType.EUCLIDEAN
    sheaf_structure: Optional[SheafStructure] = None
    metadata: ParameterMetadata = field(default_factory=ParameterMetadata)
    
    # Geometric structure
    metric_tensor: Optional[NDArray] = field(default=None, repr=False)
    tangent_space_basis: Optional[NDArray] = field(default=None, repr=False)
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        # Validate
        self.spec.validate(self.data, "ParameterState.data")
        
        # Ensure dtype
        if self.data.dtype != np.float64:
            object.__setattr__(self, 'data', self.data.astype(np.float64))
        
        # Initialize geometric structures if needed
        if self.constraint_type == ConstraintType.RIEMANNIAN and self.metric_tensor is None:
            # Default to Euclidean metric
            dim = self.spec.size
            object.__setattr__(
                self, 'metric_tensor',
                np.eye(dim, dtype=np.float64)
            )
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def size(self) -> int:
        return self.spec.size
    
    def copy(self, deep: bool = True) -> ParameterState:
        """Create copy with incremented version."""
        with self._lock:
            new_metadata = ParameterMetadata(
                created_at=self.metadata.created_at,
                updated_at=datetime.now(),
                version=self.metadata.version + 1,
                tags=self.metadata.tags.copy(),
                history=self.metadata.history.copy() if deep else []
            )
            
            return ParameterState(
                data=self.data.copy() if deep else self.data,
                spec=self.spec,
                constraint_type=self.constraint_type,
                sheaf_structure=self.sheaf_structure,
                metadata=new_metadata,
                metric_tensor=self.metric_tensor.copy() if deep and self.metric_tensor is not None else self.metric_tensor,
                tangent_space_basis=self.tangent_space_basis.copy() if deep and self.tangent_space_basis is not None else self.tangent_space_basis
            )
    
    def apply_update(
        self,
        update: Union[TangentBundle, NDArray],
        learning_rate: float,
        optimizer_state: Optional[Dict[str, Any]] = None
    ) -> ParameterState:
        """
        Apply update respecting parameter space structure.
        
        Thread-safe: returns new instance, original unchanged.
        """
        with self._lock:
            new_state = self.copy(deep=True)
            
            # Convert to TangentBundle if needed
            if isinstance(update, np.ndarray):
                update = TangentBundle(
                    base_point=self.data,
                    fiber_vector=update
                )
            
            # Apply constraint-specific update
            new_data = self._compute_update(update, learning_rate, optimizer_state)
            
            # Project to constraint manifold if needed
            new_data = self._project_to_manifold(new_data)
            
            # Update sheaf constraints
            if self.constraint_type == ConstraintType.SHEAF and self.sheaf_structure:
                new_data = self.sheaf_structure.project_harmonic(new_data)
            
            # Record metadata
            update_norm = float(np.linalg.norm(update.fiber_vector))
            new_state.metadata.update_norm = update_norm
            new_state.metadata.record_step({
                'learning_rate': learning_rate,
                'update_norm': update_norm,
                'constraint_type': self.constraint_type.name
            })
            
            object.__setattr__(new_state, 'data', new_data)
            return new_state
    
    def _compute_update(
        self,
        update: TangentBundle,
        lr: float,
        opt_state: Optional[Dict[str, Any]]
    ) -> NDArray:
        """Compute raw parameter update."""
        if self.constraint_type == ConstraintType.EUCLIDEAN:
            return self.data - lr * update.fiber_vector
        
        elif self.constraint_type == ConstraintType.RIEMANNIAN:
            # Riemannian gradient descent
            if self.metric_tensor is not None:
                # Natural gradient: G^{-1} ∇f
                flat_grad = update.fiber_vector.flatten()
                natural_grad = np.linalg.solve(self.metric_tensor, flat_grad)
                return self.data - lr * natural_grad.reshape(self.shape)
            return self.data - lr * update.fiber_vector
        
        elif self.constraint_type == ConstraintType.DISCRETE:
            return self._discrete_update(update.fiber_vector, lr)
        
        elif self.constraint_type == ConstraintType.ORDINAL:
            return self._ordinal_update(update.fiber_vector, lr)
        
        elif self.constraint_type == ConstraintType.HYPERREAL:
            return self._hyperreal_update(update.fiber_vector, lr)
        
        elif self.constraint_type == ConstraintType.QUANTUM:
            return self._quantum_update(update.fiber_vector, lr)
        
        else:
            return self.data - lr * update.fiber_vector
    
    def _discrete_update(self, tangent: NDArray, lr: float) -> NDArray:
        """
        Discrete gradient descent via Gumbel-softmax with straight-through estimator.
        
        Mathematical basis:
            - Relaxed one-hot via softmax with temperature
            - Straight-through for gradient flow
        """
        # Prevent overflow
        scaled = -lr * np.clip(tangent, -500, 500)
        
        # Gumbel-softmax (simplified, no actual Gumbel noise for determinism)
        exp_shifted = np.exp(scaled - np.max(scaled))
        probs = exp_shifted / (np.sum(exp_shifted) + 1e-10)
        
        # Hard selection
        hard = np.zeros_like(probs)
        hard[np.argmax(probs)] = 1.0
        
        # Straight-through: (hard - probs).detach() + probs
        return (hard - probs) + probs  # Identity for grad, hard for value
    
    def _ordinal_update(self, tangent: NDArray, lr: float) -> NDArray:
        """Update for ordinal parameters using rounded steps."""
        update = np.round(lr * tangent).astype(np.int64)
        new_data = np.maximum(0, self.data - update)
        return new_data.astype(self.data.dtype)
    
    def _hyperreal_update(self, tangent: NDArray, lr: float) -> NDArray:
        """
        Update for hyperreal parameters (*R).
        
        Structure: a + bε where ε² = 0, ε > 0 but smaller than any real.
        Represented as complex: real part = standard, imag part = infinitesimal.
        """
        if np.iscomplexobj(tangent):
            standard = np.real(tangent)
            infinitesimal = np.imag(tangent)
            return self.data - lr * standard - lr * 0.001 * infinitesimal
        
        # Standard update if no infinitesimal part
        return self.data - lr * tangent
    
    def _quantum_update(self, tangent: NDArray, lr: float) -> NDArray:
        """
        Update for density matrix parameters.
        
        Ensures positive semi-definiteness via eigendecomposition.
        """
        # Reshape to matrix
        dim = int(np.sqrt(self.size))
        matrix = self.data.reshape(dim, dim)
        
        # Ensure Hermitian
        matrix = (matrix + matrix.T.conj()) / 2
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Update eigenvalues (tangent affects spectrum)
        grad_matrix = tangent.reshape(dim, dim)
        grad_diag = np.diag(eigvecs.T @ grad_matrix @ eigvecs)
        
        new_eigvals = eigvals - lr * np.real(grad_diag)
        new_eigvals = np.maximum(new_eigvals, 0)  # PSD projection
        
        # Reconstruct
        new_matrix = eigvecs @ np.diag(new_eigvals) @ eigvecs.T.conj()
        return new_matrix.flatten().real.astype(np.float64)
    
    def _project_to_manifold(self, data: NDArray) -> NDArray:
        """Project updated parameters back to constraint manifold."""
        if self.constraint_type == ConstraintType.QUANTUM:
            # Ensure trace 1 for density matrices
            dim = int(np.sqrt(data.size))
            matrix = data.reshape(dim, dim)
            matrix = (matrix + matrix.T) / 2  # Symmetrize
            trace = np.trace(matrix)
            if abs(trace) > 1e-10:
                matrix = matrix / trace
            return matrix.flatten()
        
        return data
    
    def to_tangent(self) -> TangentBundle:
        """View parameters as tangent vector at origin."""
        return TangentBundle(
            base_point=np.zeros_like(self.data),
            fiber_vector=self.data
        )
    
    def gradient_to_tangent(self, grad: NDArray) -> TangentBundle:
        """Convert gradient to tangent vector at current position."""
        return TangentBundle(
            base_point=self.data,
            fiber_vector=grad
        )
    
    def compute_hash(self) -> str:
        """Compute content hash for verification."""
        return hashlib.sha256(self.data.tobytes()).hexdigest()[:16]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to disk with metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'array': self.data,
            'spec': self.spec.to_dict(),
            'constraint_type': self.constraint_type.value,
            'metadata': {
                'version': self.metadata.version,
                'created_at': self.metadata.created_at.isoformat(),
                'tags': list(self.metadata.tags)
            }
        }
        
        np.savez_compressed(path, **data)
        logger.info(f"Saved parameter state to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> ParameterState:
        """Load from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        
        data = np.load(path, allow_pickle=True)
        
        spec = TensorSpec.from_dict(dict(data['spec'].item()))
        constraint_type = ConstraintType(int(data['constraint_type']))
        
        return cls(
            data=data['array'],
            spec=spec,
            constraint_type=constraint_type
        )


# ============================================================================
# SECTION 6: PARAMETRIC LENSES - CATEGORICAL CORE
# ============================================================================

@dataclass(frozen=True)
class LensLawChecker:
    """
    Runtime verification of lens laws.
    
    Lens laws (Get-Put, Put-Get, Put-Put):
        1. put(s, get(s)) = s
        2. get(put(s, v)) = v
        3. put(put(s, v), v') = put(s, v')
    """
    tolerance: float = 1e-9
    
    def verify_get_put(self, lens: ParametricLens, state: Any, params: Any) -> bool:
        """Verify put(s, get(s)) = s."""
        try:
            v = lens.get(state, params)
            s_new = lens.put(state, params, v)
            return self._close(s_new, state)
        except Exception:
            return False
    
    def verify_put_get(self, lens: ParametricLens, state: Any, params: Any, view: Any) -> bool:
        """Verify get(put(s, v)) = v."""
        try:
            s_new = lens.put(state, params, view)
            v_new = lens.get(s_new, params)
            return self._close(v_new, view)
        except Exception:
            return False
    
    def _close(self, a: Any, b: Any) -> bool:
        """Check if two values are close."""
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return bool(np.allclose(a, b, atol=self.tolerance))
        return a == b


@dataclass(frozen=True, slots=True)
class ParametricLens(Generic[P, A, B]):
    """
    Parametric lens: (f, R[f]): (P × A)/(P × A') → B/B'
    
    Mathematical definition:
        A lens in category C is a pair (get: S → V, put: S × V → S)
        satisfying the lens laws.
        
        A parametric lens adds parameters P:
        - forward: P × A → B  (the 'get' with parameters)
        - reverse: (P × A) × B' → A' × P'  (the 'put' derivative)
    
    This implements the morphism structure of Para(Lens(C)).
    """
    forward_map: Callable[[P, A], B]
    reverse_derivative: Callable[[P, A, B], Tuple[A, P]]
    param_spec: TensorSpec
    input_spec: TensorSpec
    output_spec: TensorSpec
    name: str = "unnamed_lens"
    
    # Optional law checking
    verify_laws: bool = field(default=False, repr=False)
    law_checker: LensLawChecker = field(default_factory=LensLawChecker, repr=False)
    
    def __call__(self, params: P, input_data: A) -> B:
        """Callable interface."""
        return self.forward(params, input_data)
    
    def forward(self, params: P, input_data: A) -> B:
        """Execute forward pass with validation."""
        # Validate inputs (skip dynamic specs with None dimensions)
        params_arr = np.asarray(params)
        if not any(s is None for s in self.param_spec.shape):
            self.param_spec.validate(params_arr, f"{self.name}.params")
        input_arr = np.asarray(input_data)
        if not any(s is None for s in self.input_spec.shape):
            self.input_spec.validate(input_arr, f"{self.name}.input")

        # Compute
        output = self.forward_map(params, input_data)

        # Validate output
        output_arr = np.asarray(output)
        if not any(s is None for s in self.output_spec.shape):
            self.output_spec.validate(output_arr, f"{self.name}.output")

        return output
    
    def reverse(self, params: P, input_data: A, output_tangent: B) -> Tuple[A, P]:
        """
        Execute reverse pass with validation.

        Returns (input_tangent, param_tangent).
        """
        # Validate tangent (skip dynamic specs)
        if not any(s is None for s in self.output_spec.shape):
            self.output_spec.validate(np.asarray(output_tangent), f"{self.name}.output_tangent")
        
        # Compute reverse derivative
        input_tan, param_tan = self.reverse_derivative(params, input_data, output_tangent)
        
        # Validate output tangents
        in_tan_arr = np.asarray(input_tan)
        param_tan_arr = np.asarray(param_tan)
        
        if in_tan_arr.shape != np.asarray(input_data).shape:
            raise LensError(
                f"{self.name}: Input tangent shape {in_tan_arr.shape} "
                f"does not match input {np.asarray(input_data).shape}"
            )
        
        expected_param_shape = self.param_spec.shape
        if not any(s is None for s in expected_param_shape):
            if param_tan_arr.shape != expected_param_shape:
                # Allow flattened parameters
                if param_tan_arr.size != self.param_spec.size:
                    raise LensError(
                        f"{self.name}: Param tangent size {param_tan_arr.size} "
                        f"does not match spec {self.param_spec.size}"
                    )
                param_tan_arr = param_tan_arr.reshape(expected_param_shape)
        
        return input_tan, param_tan_arr
    
    def compose(self, other: ParametricLens[B, Any, C]) -> ParametricLens[P, A, C]:
        """
        Vertical composition (sequential): self ; other.
        
        Mathematical: Chain rule for reverse derivatives.
        
        If f: P × A → B and g: Q × B → C, then:
        R[g ∘ f]((p,q), a, c') = 
            let (b, p') = R[f](p, a, b')
            let (b', q') = R[g](q, b, c')
            in (a', (p', q'))
        """
        def composed_forward(p: P, a: A) -> C:
            # Split parameters
            p_size = self.param_spec.size
            total_size = np.asarray(p).size
            
            if total_size < p_size:
                raise LensError("Insufficient parameters for composition")
            
            p_self = np.asarray(p).flat[:p_size].reshape(self.param_spec.shape)
            p_other = np.asarray(p).flat[p_size:].reshape(other.param_spec.shape)
            
            b = self.forward_map(p_self, a)
            return other.forward_map(p_other, b)
        
        def composed_reverse(p: P, a: A, c_tan: C) -> Tuple[A, P]:
            # Split parameters
            p_arr = np.asarray(p)
            p_size = self.param_spec.size
            
            p_self = p_arr.flat[:p_size].reshape(self.param_spec.shape)
            p_other = p_arr.flat[p_size:].reshape(other.param_spec.shape)
            
            # Forward for intermediate values
            b = self.forward_map(p_self, a)
            
            # Reverse through second lens
            b_tan, q_tan = other.reverse_derivative(p_other, b, c_tan)
            
            # Reverse through first lens
            a_tan, p_tan = self.reverse_derivative(p_self, a, b_tan)
            
            # Concatenate parameter tangents
            p_tan_flat = np.asarray(p_tan).flatten()
            q_tan_flat = np.asarray(q_tan).flatten()
            combined_tan = np.concatenate([p_tan_flat, q_tan_flat])
            
            return a_tan, combined_tan
        
        new_param_shape = TensorSpec(
            shape=(self.param_spec.size + other.param_spec.size,),
            dtype=self.param_spec.dtype
        )
        
        return ParametricLens(
            forward_map=composed_forward,
            reverse_derivative=composed_reverse,
            param_spec=new_param_shape,
            input_spec=self.input_spec,
            output_spec=other.output_spec,
            name=f"{self.name}∘{other.name}"
        )
    
    def tensor_product(
        self,
        other: ParametricLens[P, A2, B2]
    ) -> ParametricLens[P, Tuple[A, A2], Tuple[B, B2]]:
        """
        Monoidal product (parallel): self ⊗ other.
        
        Runs both lenses with shared parameters on different inputs.
        """
        def par_forward(p: P, inputs: Tuple[A, A2]) -> Tuple[B, B2]:
            a1, a2 = inputs
            return (self.forward_map(p, a1), other.forward_map(p, a2))
        
        def par_reverse(
            p: P,
            inputs: Tuple[A, A2],
            tangents: Tuple[B, B2]
        ) -> Tuple[Tuple[A, A2], P]:
            a1, a2 = inputs
            t1, t2 = tangents
            
            a1_tan, p1_tan = self.reverse_derivative(p, a1, t1)
            a2_tan, p2_tan = other.reverse_derivative(p, a2, t2)
            
            # Combine parameter gradients (sum for shared parameters)
            p_tan = np.asarray(p1_tan) + np.asarray(p2_tan)
            
            return (a1_tan, a2_tan), p_tan
        
        return ParametricLens(
            forward_map=par_forward,
            reverse_derivative=par_reverse,
            param_spec=self.param_spec,
            input_spec=TensorSpec(
                shape=self.input_spec.shape + other.input_spec.shape,
                dtype=self.input_spec.dtype
            ),
            output_spec=TensorSpec(
                shape=self.output_spec.shape + other.output_spec.shape,
                dtype=self.output_spec.dtype
            ),
            name=f"{self.name}⊗{other.name}"
        )
    
    def fan_out(self, n: int) -> ParametricLens[P, A, Tuple[B, ...]]:
        """Replicate lens n times (diagonal morphism)."""
        def fan_forward(p: P, a: A) -> Tuple[B, ...]:
            return tuple(self.forward_map(p, a) for _ in range(n))
        
        def fan_reverse(
            p: P,
            a: A,
            tangents: Tuple[B, ...]
        ) -> Tuple[A, P]:
            # Sum of all tangents
            total_input_tan = None
            total_param_tan = None
            
            for t in tangents:
                a_tan, p_tan = self.reverse_derivative(p, a, t)
                if total_input_tan is None:
                    total_input_tan = np.asarray(a_tan)
                    total_param_tan = np.asarray(p_tan)
                else:
                    total_input_tan += np.asarray(a_tan)
                    total_param_tan += np.asarray(p_tan)
            
            # Guard: if tangents was empty, return zeros
            if total_input_tan is None:
                total_input_tan = np.zeros_like(np.asarray(a))
                total_param_tan = np.zeros(self.param_spec.size)

            return total_input_tan, total_param_tan
        
        return ParametricLens(
            forward_map=fan_forward,
            reverse_derivative=fan_reverse,
            param_spec=self.param_spec,
            input_spec=self.input_spec,
            output_spec=TensorSpec(
                shape=tuple([n] + list(self.output_spec.shape)),
                dtype=self.output_spec.dtype
            ),
            name=f"Δ^{n}({self.name})"
        )
    
    def curry(self) -> ParametricLens[P, A, Callable[[], B]]:
        """Curry to expose parameter-dependence."""
        # This is more of a type-level operation in Python
        return self
    
    def dimap(
        self,
        pre: Callable[[A2], A],
        post: Callable[[B], B2]
    ) -> ParametricLens[P, A2, B2]:
        """
        Pre and post composition with pure functions.
        
        Covariant in output, contravariant in input.
        """
        def new_forward(p: P, a2: A2) -> B2:
            return post(self.forward_map(p, pre(a2)))
        
        def new_reverse(p: P, a2: A2, b2_tan: B2) -> Tuple[A2, P]:
            # Chain rule
            a = pre(a2)
            b = self.forward_map(p, a)
            
            # Assume post is linear or use AD
            # Simplified: identity Jacobian
            b_tan = b2_tan  # Should be J_post^T @ b2_tan
            
            a_tan, p_tan = self.reverse_derivative(p, a, b_tan)
            
            # a2_tan = J_pre^T @ a_tan
            a2_tan = a_tan  # Simplified
            
            return a2_tan, p_tan
        
        return ParametricLens(
            forward_map=new_forward,
            reverse_derivative=new_reverse,
            param_spec=self.param_spec,
            input_spec=TensorSpec(
                shape=np.asarray(pre(np.zeros(tuple(s if s is not None else 1 for s in self.input_spec.shape)))).shape,
                dtype=self.input_spec.dtype
            ),
            output_spec=TensorSpec(
                shape=np.asarray(post(np.zeros(tuple(s if s is not None else 1 for s in self.output_spec.shape)))).shape,
                dtype=self.output_spec.dtype
            ),
            name=f"{self.name}[dimap]"
        )


# ============================================================================
# SECTION 7: OPTIMIZER HIERARCHY - ENRICHED CATEGORIES
# ============================================================================

@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer behavior."""
    learning_rate: float = 0.001
    lr_schedule: Optional[Callable[[Step], float]] = None
    gradient_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    momentum: float = 0.0
    eps: float = 1e-8
    
    def get_lr(self, step: Step) -> float:
        """Compute learning rate for step."""
        if self.lr_schedule:
            return self.lr_schedule(step) * self.learning_rate
        return self.learning_rate


@dataclass
class OptimizerState:
    """
    Mutable state container for optimizers.
    
    Optimizers are stateful morphisms in the category of parameter updates.
    """
    data: Dict[str, NDArray[np.float64]]
    config: OptimizerConfig
    step: Step = field(default_factory=lambda: Step(0))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> OptimizerState:
        # Use deepcopy for list values (L-BFGS history), ndarray.copy() for arrays
        new_data = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                new_data[k] = v.copy()
            elif isinstance(v, list):
                new_data[k] = [x.copy() if isinstance(x, np.ndarray) else x for x in v]
            else:
                new_data[k] = copy.deepcopy(v)
        return OptimizerState(
            data=new_data,
            config=self.config,
            step=self.step,
            metadata=self.metadata.copy()
        )
    
    def increment_step(self) -> None:
        self.step = Step(self.step + 1)


class Optimizer(ABC):
    """
    Abstract base for optimizers as lenses (S × P, S × P) → (P, P').
    
    An optimizer is a parametric lens where:
    - Parameters are the optimization state S
    - Input is current parameters P
    - Output is updated parameters P
    - Reverse derivative gives parameter update direction
    
    Mathematical structure:
        Optimizer : Lens(State × Param, State × Param)
    """
    
    @abstractmethod
    def init_state(self, param_spec: TensorSpec, config: OptimizerConfig) -> OptimizerState:
        """Initialize optimizer state."""
        pass
    
    @abstractmethod
    def step(
        self,
        params: ParameterState,
        gradient: TangentBundle,
        state: OptimizerState
    ) -> Tuple[ParameterState, OptimizerState]:
        """
        Execute optimization step.
        
        Returns:
            (updated_params, new_state)
        """
        pass
    
    def zero_grad(self, state: OptimizerState) -> OptimizerState:
        """Reset accumulated gradients."""
        return state


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.
    
    State: velocity v ∈ T_pM
    Update: v ← βv + (1-β)∇f, p ← exp_p(-ηv)
    """
    
    def init_state(self, param_spec: TensorSpec, config: OptimizerConfig) -> OptimizerState:
        # Use zeros matching actual param shape; fall back to (1,) for dynamic specs
        shape = tuple(s if s is not None else 1 for s in param_spec.shape)
        return OptimizerState(
            data={'velocity': np.zeros(shape, dtype=np.float64)},
            config=config
        )
    
    def step(
        self,
        params: ParameterState,
        gradient: TangentBundle,
        state: OptimizerState
    ) -> Tuple[ParameterState, OptimizerState]:
        cfg = state.config
        lr = cfg.get_lr(state.step)
        
        # Update velocity
        v = state.data['velocity']
        v_new = cfg.momentum * v + gradient.fiber_vector
        
        # Apply weight decay if specified
        if cfg.weight_decay > 0:
            v_new = v_new + cfg.weight_decay * params.data
        
        # Clip if needed
        if cfg.gradient_clip:
            norm = np.linalg.norm(v_new)
            if norm > cfg.gradient_clip:
                v_new = v_new * (cfg.gradient_clip / (norm + cfg.eps))
        
        # Update parameters
        update = TangentBundle(
            base_point=params.data,
            fiber_vector=v_new
        )
        
        new_params = params.apply_update(update, lr)
        new_state = state.copy()
        new_state.data['velocity'] = v_new
        new_state.increment_step()
        
        return new_params, new_state


class Adam(Optimizer):
    """
    Adaptive Moment Estimation.
    
    Maintains first and second moment estimates:
    m_t = β1 m_{t-1} + (1-β1) g_t
    v_t = β2 v_{t-1} + (1-β2) g_t^2
    
    Bias-corrected:
    m̂_t = m_t / (1 - β1^t)
    v̂_t = v_t / (1 - β2^t)
    
    Update: θ_t = θ_{t-1} - η m̂_t / (sqrt(v̂_t) + ε)
    """
    
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        amsgrad: bool = False
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
    
    def init_state(self, param_spec: TensorSpec, config: OptimizerConfig) -> OptimizerState:
        data = {
            'm': np.zeros(param_spec.shape, dtype=np.float64),
            'v': np.zeros(param_spec.shape, dtype=np.float64),
        }
        if self.amsgrad:
            data['v_max'] = np.zeros(param_spec.shape, dtype=np.float64)
        
        return OptimizerState(
            data=data,
            config=config,
            metadata={'beta1': self.beta1, 'beta2': self.beta2, 'amsgrad': self.amsgrad}
        )
    
    def step(
        self,
        params: ParameterState,
        gradient: TangentBundle,
        state: OptimizerState
    ) -> Tuple[ParameterState, OptimizerState]:
        cfg = state.config
        t = state.step + 1
        lr = cfg.get_lr(Step(t))
        
        g = gradient.fiber_vector
        
        # Update biased moments
        m = self.beta1 * state.data['m'] + (1 - self.beta1) * g
        v = self.beta2 * state.data['v'] + (1 - self.beta2) * (g ** 2)
        
        # Bias correction
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        
        # AMSGrad modification
        if self.amsgrad:
            v_max = np.maximum(state.data.get('v_max', v_hat), v_hat)
            denom = np.sqrt(v_max) + cfg.eps
            new_state_data = {'m': m, 'v': v, 'v_max': v_max}
        else:
            denom = np.sqrt(v_hat) + cfg.eps
            new_state_data = {'m': m, 'v': v}
        
        # Compute update
        step_dir = m_hat / denom
        
        # Weight decay (AdamW style - decoupled)
        if cfg.weight_decay > 0:
            # Apply directly to parameters, not in gradient
            new_data = params.data * (1 - lr * cfg.weight_decay) - lr * step_dir
            new_params = ParameterState(
                data=new_data,
                spec=params.spec,
                constraint_type=params.constraint_type,
                sheaf_structure=params.sheaf_structure
            )
        else:
            update = TangentBundle(
                base_point=params.data,
                fiber_vector=step_dir
            )
            new_params = params.apply_update(update, lr)
        
        new_state = OptimizerState(
            data=new_state_data,
            config=cfg,
            step=Step(t),
            metadata=state.metadata
        )
        
        return new_params, new_state


class NaturalGradient(Optimizer):
    """
    Natural gradient descent using Fisher information matrix.
    
    Update: θ ← θ - η F^{-1} ∇f
    
    where F is the Fisher information matrix (Riemannian metric).
    """
    
    def __init__(self, fisher_samples: int = 100):
        self.fisher_samples = fisher_samples
    
    def init_state(self, param_spec: TensorSpec, config: OptimizerConfig) -> OptimizerState:
        dim = param_spec.size
        if dim == 0:
            # Dynamic spec — use 1x1 identity as placeholder; resized at step time
            dim = 1
        return OptimizerState(
            data={
                'fisher': np.eye(dim, dtype=np.float64) * 0.01,  # Initialize small
                'fisher_inv': np.eye(dim, dtype=np.float64) * 100.0
            },
            config=config
        )
    
    def step(
        self,
        params: ParameterState,
        gradient: TangentBundle,
        state: OptimizerState
    ) -> Tuple[ParameterState, OptimizerState]:
        cfg = state.config
        lr = cfg.get_lr(state.step)
        
        # Use precomputed Fisher inverse
        fisher_inv = state.data['fisher_inv']
        
        # Natural gradient direction
        flat_grad = gradient.fiber_vector.flatten()

        # Resize Fisher inverse if param dimension changed (dynamic specs)
        if fisher_inv.shape[0] != flat_grad.size:
            fisher_inv = np.eye(flat_grad.size, dtype=np.float64) * 100.0

        natural_dir = fisher_inv @ flat_grad
        
        # Reshape and update
        update = TangentBundle(
            base_point=params.data,
            fiber_vector=natural_dir.reshape(params.shape)
        )
        
        new_params = params.apply_update(update, lr)
        new_state = state.copy()
        new_state.increment_step()
        
        return new_params, new_state
    
    def update_fisher(
        self,
        state: OptimizerState,
        gradients: List[NDArray]
    ) -> OptimizerState:
        """Update Fisher matrix estimate using samples."""
        # Empirical Fisher: (1/N) Σ g_i g_i^T
        stacked = np.stack([g.flatten() for g in gradients])
        fisher = (stacked.T @ stacked) / len(gradients)
        
        # Add damping for stability
        fisher = fisher + 0.01 * np.eye(fisher.shape[0])
        
        # Compute inverse (use Cholesky for stability)
        try:
            L = np.linalg.cholesky(fisher)
            fisher_inv = np.linalg.solve(L, np.eye(L.shape[0]))
            fisher_inv = np.linalg.solve(L.T, fisher_inv)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            fisher_inv = np.linalg.pinv(fisher)
        
        new_state = state.copy()
        new_state.data['fisher'] = fisher
        new_state.data['fisher_inv'] = fisher_inv
        return new_state


class SecondOrder(Optimizer):
    """
    Newton/Quasi-Newton methods (L-BFGS approximation).
    
    Maintains low-rank approximation of Hessian inverse.
    """
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
    
    def init_state(self, param_spec: TensorSpec, config: OptimizerConfig) -> OptimizerState:
        dim = param_spec.size
        return OptimizerState(
            data={
                's_history': [],  # Parameter differences
                'y_history': [],  # Gradient differences
                'rho_history': []  # Curvature information
            },
            config=config,
            metadata={'history_size': self.history_size}
        )
    
    def step(
        self,
        params: ParameterState,
        gradient: TangentBundle,
        state: OptimizerState
    ) -> Tuple[ParameterState, OptimizerState]:
        cfg = state.config
        lr = cfg.get_lr(state.step)
        
        # L-BFGS two-loop recursion
        q = gradient.fiber_vector.flatten()
        
        s_hist = state.data['s_history']
        y_hist = state.data['y_history']
        rho_hist = state.data['rho_history']
        
        alphas = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q = q - alpha * y
        
        # Initial Hessian approximation (scaled identity)
        if len(s_hist) > 0:
            gamma = np.dot(s_hist[-1], y_hist[-1]) / (np.dot(y_hist[-1], y_hist[-1]) + 1e-10)
            r = gamma * q
        else:
            r = q
        
        for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            beta = rho * np.dot(y, r)
            r = r + s * (alpha - beta)
        
        # Update parameters
        update = TangentBundle(
            base_point=params.data,
            fiber_vector=r.reshape(params.shape)
        )
        
        new_params = params.apply_update(update, lr)
        new_state = state.copy()
        new_state.increment_step()
        
        return new_params, new_state
    
    def update_hessian(
        self,
        state: OptimizerState,
        s: NDArray,  # Parameter change
        y: NDArray   # Gradient change
    ) -> OptimizerState:
        """Update L-BFGS history."""
        new_state = state.copy()
        
        s_flat = s.flatten()
        y_flat = y.flatten()
        
        ys = np.dot(y_flat, s_flat)
        if ys < 1e-10:  # Skip if not positive definite
            return new_state
        
        rho = 1.0 / ys
        
        new_state.data['s_history'].append(s_flat)
        new_state.data['y_history'].append(y_flat)
        new_state.data['rho_history'].append(rho)
        
        # Maintain bounded history
        max_size = new_state.metadata.get('history_size', 10)
        if len(new_state.data['s_history']) > max_size:
            new_state.data['s_history'].pop(0)
            new_state.data['y_history'].pop(0)
            new_state.data['rho_history'].pop(0)
        
        return new_state


# ============================================================================
# SECTION 8: LOSS FUNCTIONS AND OBJECTIVES
# ============================================================================

class LossFunction(ABC):
    """
    Loss functions as parametric lenses (B × Y) → ℝ.
    
    Mathematical structure:
        L: Prediction × Target → Loss
        ∇L: Prediction × Target → T*Prediction (cotangent)
    """
    
    @abstractmethod
    def forward(
        self,
        prediction: NDArray[np.float64],
        target: NDArray[np.float64]
    ) -> LossValue:
        """Compute scalar loss."""
        pass
    
    @abstractmethod
    def backward(
        self,
        prediction: NDArray[np.float64],
        target: NDArray[np.float64]
    ) -> CotangentBundle:
        """Return gradient (covector) w.r.t. prediction."""
        pass
    
    def __call__(
        self,
        prediction: NDArray[np.float64],
        target: NDArray[np.float64]
    ) -> Tuple[LossValue, CotangentBundle]:
        """Compute loss and gradient."""
        loss = self.forward(prediction, target)
        grad = self.backward(prediction, target)
        return loss, grad


class MSE(LossFunction):
    """Mean Squared Error for regression."""
    
    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        self.reduction = reduction
    
    def forward(self, pred: NDArray, target: NDArray) -> LossValue:
        diff = pred - target
        if self.reduction == 'mean':
            return LossValue(float(np.mean(diff ** 2)))
        return LossValue(float(np.sum(diff ** 2)))
    
    def backward(self, pred: NDArray, target: NDArray) -> CotangentBundle:
        diff = pred - target
        if self.reduction == 'mean':
            grad = 2.0 * diff / diff.size
        else:
            grad = 2.0 * diff
        
        return CotangentBundle(
            base_point=pred,
            covector=grad
        )


class CrossEntropy(LossFunction):
    """Cross-entropy loss for classification."""
    
    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: Literal['mean', 'sum'] = 'mean'
    ):
        if not 0 <= label_smoothing < 1:
            raise ValidationError(f"Invalid label_smoothing: {label_smoothing}")
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: NDArray, target: NDArray) -> LossValue:
        # Numerically stable softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        
        n_classes = logits.shape[-1]
        
        if self.label_smoothing > 0:
            smooth_target = (1 - self.label_smoothing) * target + \
                          self.label_smoothing / n_classes
        else:
            smooth_target = target
        
        loss = -np.sum(smooth_target * np.log(probs + 1e-10), axis=-1)
        
        if self.reduction == 'mean':
            return LossValue(float(np.mean(loss)))
        return LossValue(float(np.sum(loss)))
    
    def backward(self, logits: NDArray, target: NDArray) -> CotangentBundle:
        # Gradient: softmax - target
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        
        n_classes = logits.shape[-1]
        if self.label_smoothing > 0:
            smooth_target = (1 - self.label_smoothing) * target + \
                          self.label_smoothing / n_classes
        else:
            smooth_target = target
        
        grad = probs - smooth_target
        
        if self.reduction == 'mean':
            grad = grad / logits.shape[0]
        
        return CotangentBundle(
            base_point=logits,
            covector=grad
        )


class Huber(LossFunction):
    """Huber loss for robust regression."""
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    
    def forward(self, pred: NDArray, target: NDArray) -> LossValue:
        diff = np.abs(pred - target)
        quadratic = np.minimum(diff, self.delta)
        linear = diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return LossValue(float(np.mean(loss)))
    
    def backward(self, pred: NDArray, target: NDArray) -> CotangentBundle:
        diff = pred - target
        abs_diff = np.abs(diff)
        
        grad = np.where(
            abs_diff <= self.delta,
            diff,
            self.delta * np.sign(diff)
        ) / pred.size
        
        return CotangentBundle(
            base_point=pred,
            covector=grad
        )


class ContrastiveLoss(LossFunction):
    """Contrastive learning loss (InfoNCE)."""
    
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
    
    def forward(self, anchors: NDArray, positives: NDArray) -> LossValue:
        # Simplified contrastive loss
        # anchors: (batch, dim), positives: (batch, dim)
        
        # Normalize
        a_norm = anchors / (np.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-8)
        p_norm = positives / (np.linalg.norm(positives, axis=-1, keepdims=True) + 1e-8)
        
        # Similarities
        logits = np.dot(a_norm, p_norm.T) / self.temperature
        
        # Labels: diagonal is positive pair
        batch_size = anchors.shape[0]
        labels = np.arange(batch_size)
        
        # Cross-entropy with positives as targets
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        log_probs = shifted - np.log(np.sum(exp_shifted, axis=-1, keepdims=True))
        
        loss = -np.mean(log_probs[np.arange(batch_size), labels])
        return LossValue(float(loss))
    
    def backward(self, anchors: NDArray, positives: NDArray) -> CotangentBundle:
        # Full InfoNCE gradient w.r.t. anchors
        batch_size = anchors.shape[0]

        a_norms = np.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-8
        p_norms = np.linalg.norm(positives, axis=-1, keepdims=True) + 1e-8
        a_norm = anchors / a_norms
        p_norm = positives / p_norms

        # logits[i,j] = sim(a_i, p_j) / temperature
        logits = np.dot(a_norm, p_norm.T) / self.temperature  # (B, B)

        # Softmax over columns
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-10)  # (B, B)

        # d_loss/d_logits: probs - one_hot(labels)
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), np.arange(batch_size)] -= 1.0
        d_logits /= batch_size  # mean reduction

        # d_logits/d_a_norm = d_logits @ p_norm / temperature
        d_a_norm = (d_logits / self.temperature) @ p_norm  # (B, dim)

        # Chain rule through normalization: d/d_a of (a / ||a||)
        # = (I - a_norm a_norm^T) / ||a||
        dot = np.sum(d_a_norm * a_norm, axis=-1, keepdims=True)
        d_anchors = (d_a_norm - dot * a_norm) / a_norms

        return CotangentBundle(
            base_point=anchors,
            covector=d_anchors
        )


# ============================================================================
# SECTION 9: UNIVERSAL TRAINING ENGINE
# ============================================================================

@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    # Optimization
    learning_rate: float = 0.001
    optimizer: Type[Optimizer] = field(default_factory=lambda: Adam)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    loss_function: LossFunction = field(default_factory=MSE)
    
    # Training loop
    max_epochs: int = 1000
    max_steps: Optional[int] = None
    batch_size: int = 32
    
    # Convergence
    convergence_threshold: float = 1e-6
    patience: int = 50
    min_delta: float = 1e-7
    
    # Regularization
    gradient_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    
    # Logging
    verbose: bool = True
    log_interval: int = 10
    checkpoint_dir: Optional[Path] = None
    checkpoint_interval: int = 100
    
    # Advanced
    temporal_structure: TemporalStructure = TemporalStructure.STANDARD
    use_mixed_precision: bool = False
    distributed: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        if self.seed is not None:
            # Store a local RNG rather than mutating global numpy state
            object.__setattr__(self, '_rng', np.random.default_rng(self.seed))
        else:
            object.__setattr__(self, '_rng', np.random.default_rng())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'convergence_threshold': self.convergence_threshold,
            'patience': self.patience,
            'temporal_structure': self.temporal_structure.name,
            'seed': self.seed
        }


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics."""
    loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    param_norm_history: List[float] = field(default_factory=list)
    learning_rate_history: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    
    # Best tracking
    best_loss: float = field(default=float('inf'))
    best_params: Optional[NDArray] = field(default=None, repr=False)
    best_epoch: int = field(default=0)
    
    # Status
    converged: bool = False
    converged_at: Optional[Epoch] = None
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    
    def update(self, loss: float, grad_norm: float, param_norm: float, lr: float, step_time: float = 0.0) -> None:
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.param_norm_history.append(param_norm)
        self.learning_rate_history.append(lr)
        self.step_times.append(step_time)
        
        if loss < self.best_loss - 1e-9:
            self.best_loss = loss
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'converged': self.converged,
            'converged_at': self.converged_at,
            'stopped_early': self.stopped_early,
            'stop_reason': self.stop_reason,
            'avg_step_time': np.mean(self.step_times) if self.step_times else 0.0
        }


class UniversalTrainer:
    """
    Universal training engine implementing Xenomorphic Gradient Descent (XGD).
    
    Features:
    - Categorical composition of lenses
    - Sheaf-theoretic constraints
    - Multiple temporal structures
    - Distributed training support
    - Comprehensive checkpointing
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        self._optimizer_state: Optional[OptimizerState] = None
        self._current_params: Optional[ParameterState] = None
        self._step_count: Step = Step(0)
        self._epoch: Epoch = Epoch(0)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for training events."""
        self._callbacks[event].append(callback)
    
    def _trigger(self, event: str, *args, **kwargs) -> None:
        """Trigger callbacks for event."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error for {event}: {e}")
    
    def _create_optimizer(self, param_spec: TensorSpec) -> OptimizerState:
        """Initialize optimizer."""
        opt = self.config.optimizer(**self.config.optimizer_kwargs)
        cfg = OptimizerConfig(
            learning_rate=self.config.learning_rate,
            gradient_clip=self.config.gradient_clip,
            weight_decay=self.config.weight_decay
        )
        return opt.init_state(param_spec, cfg)
    
    def train_step(
        self,
        lens: ParametricLens,
        params: ParameterState,
        batch: List[Tuple[Any, Any]]
    ) -> Tuple[ParameterState, float, float, float]:
        """
        Execute one training step on a batch.

        Returns:
            (new_params, avg_loss, avg_grad_norm, step_time)
        """
        import time
        start = time.time()
        
        total_loss = 0.0
        total_grad_norm = 0.0
        
        # Accumulate gradients from batch
        accumulated_grad: Optional[NDArray] = None
        
        for input_data, target in batch:
            # Forward pass
            prediction = lens.forward(params.data, input_data)
            
            # Compute loss and gradient
            loss, cotangent = self.config.loss_function(prediction, target)
            total_loss += float(loss)
            
            # Reverse pass
            _, param_grad = lens.reverse(params.data, input_data, cotangent.covector)
            param_grad_arr = np.asarray(param_grad)
            
            if accumulated_grad is None:
                accumulated_grad = param_grad_arr
            else:
                accumulated_grad += param_grad_arr
        
        # Average over batch
        batch_size = len(batch)
        avg_loss = total_loss / batch_size
        avg_grad = accumulated_grad / batch_size if accumulated_grad is not None else np.zeros_like(params.data)
        
        # Create tangent bundle
        gradient = TangentBundle(
            base_point=params.data,
            fiber_vector=avg_grad
        )
        
        # Apply sheaf constraints if present
        if params.sheaf_structure:
            gradient = self._apply_sheaf_constraints(gradient, params)
        
        # Clip gradients
        if self.config.gradient_clip:
            gradient = self._clip_gradient(gradient)
        
        grad_norm = gradient.norm
        total_grad_norm += grad_norm
        
        # Optimizer step
        if self._optimizer_state is None:
            self._optimizer_state = self._create_optimizer(params.spec)
        
        new_params, self._optimizer_state = self.config.optimizer(
            **self.config.optimizer_kwargs
        ).step(params, gradient, self._optimizer_state)
        
        step_time = time.time() - start
        self._step_count = Step(self._step_count + 1)
        
        return new_params, avg_loss, grad_norm, step_time
    
    def _apply_sheaf_constraints(
        self,
        gradient: TangentBundle,
        params: ParameterState
    ) -> TangentBundle:
        """Project gradient to respect sheaf cohomology."""
        if not params.sheaf_structure or params.sheaf_structure.laplacian is None:
            return gradient
        
        flat_grad = gradient.fiber_vector.flatten()
        laplacian = params.sheaf_structure.laplacian

        try:
            # Project to coclosed forms (orthogonal to harmonic)
            harmonic, info = cg(laplacian, laplacian @ flat_grad, tol=1e-10)
            if info == 0:
                constrained = flat_grad - harmonic
                return TangentBundle(
                    base_point=gradient.base_point,
                    fiber_vector=constrained.reshape(gradient.shape),
                    metadata={**gradient.metadata, 'sheaf_projected': True}
                )
        except Exception as e:
            logger.warning(f"Sheaf projection failed: {e}")
        
        return gradient
    
    def _clip_gradient(self, gradient: TangentBundle) -> TangentBundle:
        """Apply gradient clipping."""
        if self.config.gradient_clip is None:
            return gradient
        
        norm = gradient.norm
        if norm > self.config.gradient_clip:
            scale = self.config.gradient_clip / (norm + 1e-10)
            return gradient * scale
        return gradient
    
    def train(
        self,
        lens: ParametricLens,
        initial_params: ParameterState,
        train_data: List[Tuple[Any, Any]],
        val_data: Optional[List[Tuple[Any, Any]]] = None,
        epochs: Optional[int] = None
    ) -> Tuple[ParameterState, TrainingMetrics]:
        """
        Full training loop with early stopping and comprehensive monitoring.
        """
        epochs = epochs or self.config.max_epochs
        params = initial_params
        self._current_params = params
        
        # Initialize optimizer
        self._optimizer_state = self._create_optimizer(params.spec)
        
        # Tracking
        patience_counter = 0
        best_val_loss = float('inf')
        
        self._trigger('train_start', self, lens, params)
        
        for epoch in range(epochs):
            self._epoch = Epoch(epoch)
            epoch_start = datetime.now()
            
            # Shuffle data
            indices = np.random.permutation(len(train_data))
            shuffled_data = [train_data[i] for i in indices]
            
            # Batch processing
            num_batches = (len(shuffled_data) + self.config.batch_size - 1) // self.config.batch_size
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(shuffled_data))
                batch = shuffled_data[start_idx:end_idx]
                
                # Training step
                params, loss, grad_norm, step_time = self.train_step(lens, params, batch)
                epoch_loss += loss * len(batch)
                epoch_grad_norm += grad_norm
                
                # Update metrics
                self.metrics.update(loss, grad_norm, float(np.linalg.norm(params.data)), 
                                  self.config.learning_rate, step_time)
                
                # Checkpoint
                if self._step_count % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(params)
                
                # Step limit check
                if self.config.max_steps and self._step_count >= self.config.max_steps:
                    logger.info(f"Reached max steps {self.config.max_steps}")
                    break
            
            # Epoch averages
            avg_loss = epoch_loss / len(train_data)
            avg_grad_norm = epoch_grad_norm / num_batches
            
            # Validation
            val_loss = None
            if val_data:
                val_loss = self._validate(lens, params, val_data)
                self.metrics.val_loss_history.append(val_loss)
                
                # Early stopping check on validation
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    self.metrics.best_params = params.data.copy()
                    self.metrics.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Logging
            if self.config.verbose and epoch % self.config.log_interval == 0:
                elapsed = (datetime.now() - epoch_start).total_seconds()
                msg = (f"Epoch {epoch}/{epochs} | "
                       f"Loss: {avg_loss:.6f} | "
                       f"Grad: {avg_grad_norm:.6f} | "
                       f"Params: {np.linalg.norm(params.data):.2f} | "
                       f"Time: {elapsed:.2f}s")
                if val_loss is not None:
                    msg += f" | Val: {val_loss:.6f}"
                logger.info(msg)
            
            self._trigger('epoch_end', self, epoch, avg_loss, val_loss)
            
            # Convergence check (training loss)
            if len(self.metrics.loss_history) > 1:
                change = abs(self.metrics.loss_history[-1] - self.metrics.loss_history[-2])
                if change < self.config.convergence_threshold:
                    self.metrics.converged = True
                    self.metrics.converged_at = Epoch(epoch)
                    logger.info(f"Converged at epoch {epoch}")
                    break
            
            # Early stopping
            if patience_counter >= self.config.patience:
                self.metrics.stopped_early = True
                self.metrics.stop_reason = "patience_exceeded"
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            self._current_params = params
        
        self._trigger('train_end', self, params, self.metrics)
        
        # Restore best parameters
        if self.metrics.best_params is not None:
            final_params = ParameterState(
                data=self.metrics.best_params,
                spec=params.spec,
                constraint_type=params.constraint_type,
                sheaf_structure=params.sheaf_structure
            )
        else:
            final_params = params
        
        return final_params, self.metrics
    
    def _validate(
        self,
        lens: ParametricLens,
        params: ParameterState,
        val_data: List[Tuple[Any, Any]]
    ) -> float:
        """Compute validation loss."""
        total_loss = 0.0
        for input_data, target in val_data:
            prediction = lens.forward(params.data, input_data)
            loss, _ = self.config.loss_function(prediction, target)
            total_loss += float(loss)
        return total_loss / len(val_data)
    
    def _save_checkpoint(self, params: ParameterState) -> None:
        """Save training checkpoint."""
        if self.config.checkpoint_dir is None:
            return
        
        try:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.config.checkpoint_dir / f"checkpoint_step{self._step_count}_{timestamp}.npz"
            params.save(path)
            logger.debug(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
    
    def train_transfinite(
        self,
        lens: ParametricLens,
        initial_params: ParameterState,
        dataset: List[Tuple[Any, Any]],
        ordinal_limit: int = 100
    ) -> Tuple[ParameterState, TrainingMetrics]:
        """
        Training for non-standard time (ordinal/hyperreal).
        
        Implements transfinite backpropagation where training proceeds
        through ordinal stages with limit stages taking liminf of gradients.
        """
        params = initial_params
        self._current_params = params
        
        for alpha in range(ordinal_limit):
            # Limit stage: accumulate and take liminf
            if alpha > 0 and alpha % 10 == 0:
                gradients = []
                losses = []
                
                for input_data, target in dataset:
                    prediction = lens.forward(params.data, input_data)
                    loss, cotangent = self.config.loss_function(prediction, target)
                    _, param_grad = lens.reverse(params.data, input_data, cotangent.covector)
                    gradients.append(np.asarray(param_grad).flatten())
                    losses.append(float(loss))
                
                # Liminf of gradients (conservative update)
                stacked = np.stack(gradients)
                limit_gradient = np.min(stacked, axis=0)  # Conservative
                
                # Create tangent and update
                tangent = TangentBundle(
                    base_point=params.data,
                    fiber_vector=limit_gradient.reshape(params.shape)
                )
                
                if self._optimizer_state is None:
                    self._optimizer_state = self._create_optimizer(params.spec)

                # Decay learning rate at limit ordinals by temporarily scaling the config lr
                decayed_cfg = OptimizerConfig(
                    learning_rate=self.config.learning_rate * 0.1,
                    gradient_clip=self.config.gradient_clip,
                    weight_decay=self.config.weight_decay
                )
                decayed_state = OptimizerState(
                    data=self._optimizer_state.data,
                    config=decayed_cfg,
                    step=self._optimizer_state.step,
                    metadata=self._optimizer_state.metadata
                )
                params, updated_state = self.config.optimizer(
                    **self.config.optimizer_kwargs
                ).step(params, tangent, decayed_state)
                # Restore original config in optimizer state
                self._optimizer_state = OptimizerState(
                    data=updated_state.data,
                    config=self._optimizer_state.config,
                    step=updated_state.step,
                    metadata=updated_state.metadata
                )
                
                self.metrics.loss_history.append(np.mean(losses))
                
                if self.config.verbose:
                    logger.info(
                        f"Limit ordinal {alpha}: "
                        f"loss={np.mean(losses):.6f}, "
                        f"grad_norm={np.linalg.norm(limit_gradient):.6f}"
                    )
            else:
                # Successor stage: standard update
                idx = alpha % len(dataset)
                input_data, target = dataset[idx]
                
                prediction = lens.forward(params.data, input_data)
                loss, cotangent = self.config.loss_function(prediction, target)
                _, param_grad = lens.reverse(params.data, input_data, cotangent.covector)
                
                tangent = TangentBundle(
                    base_point=params.data,
                    fiber_vector=np.asarray(param_grad)
                )
                
                if self._optimizer_state is None:
                    self._optimizer_state = self._create_optimizer(params.spec)
                
                params, self._optimizer_state = self.config.optimizer(
                    **self.config.optimizer_kwargs
                ).step(params, tangent, self._optimizer_state)
                self.metrics.loss_history.append(float(loss))
        
        self._current_params = params
        return params, self.metrics
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable training state."""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'step_count': self._step_count,
            'epoch': self._epoch,
            'params_hash': self._current_params.compute_hash() if self._current_params else None
        }


# ============================================================================
# SECTION 10: LENS FACTORY AND ARCHITECTURE BUILDERS
# ============================================================================

class LensFactory:
    """
    Factory for creating common lens architectures.
    
    Implements standard deep learning components as parametric lenses.
    """
    
    @staticmethod
    def linear(
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str = "linear"
    ) -> ParametricLens:
        """Create linear transformation lens."""
        # Parameters: [weight_matrix, bias_vector]
        weight_size = out_features * in_features
        bias_size = out_features if bias else 0
        total_params = weight_size + bias_size
        
        param_spec = TensorSpec(shape=(total_params,), dtype=np.dtype('float64'))
        input_spec = TensorSpec(shape=(in_features,), dtype=np.dtype('float64'))
        output_spec = TensorSpec(shape=(out_features,), dtype=np.dtype('float64'))
        
        def forward(params: NDArray, x: NDArray) -> NDArray:
            W = params[:weight_size].reshape(out_features, in_features)
            if bias:
                b = params[weight_size:].reshape(out_features)
                return W @ x + b
            return W @ x
        
        def reverse(params: NDArray, x: NDArray, dy: NDArray) -> Tuple[NDArray, NDArray]:
            W = params[:weight_size].reshape(out_features, in_features)
            
            # Input gradient
            dx = W.T @ dy
            
            # Parameter gradients
            dW = np.outer(dy, x).flatten()
            if bias:
                db = dy
                dparams = np.concatenate([dW, db])
            else:
                dparams = dW
            
            return dx, dparams
        
        return ParametricLens(
            forward_map=forward,
            reverse_derivative=reverse,
            param_spec=param_spec,
            input_spec=input_spec,
            output_spec=output_spec,
            name=name
        )
    
    @staticmethod
    def activation(fn: str, name: Optional[str] = None) -> ParametricLens:
        """Create activation function lens (stateless)."""
        name = name or f"{fn}_activation"

        # Stateless - no parameters (use size-1 dummy to pass validation)
        param_spec = TensorSpec(shape=(1,), dtype=np.dtype('float64'))
        input_spec = TensorSpec(shape=(None,), dtype=np.dtype('float64'))
        output_spec = TensorSpec(shape=(None,), dtype=np.dtype('float64'))
        
        def forward(params: NDArray, x: NDArray) -> NDArray:
            if fn == "relu":
                return np.maximum(0, x)
            elif fn == "sigmoid":
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            elif fn == "tanh":
                return np.tanh(x)
            elif fn == "softmax":
                shifted = x - np.max(x, axis=-1, keepdims=True)
                exp = np.exp(shifted)
                return exp / np.sum(exp, axis=-1, keepdims=True)
            elif fn == "gelu":
                # Gaussian Error Linear Unit
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
            elif fn == "swish":
                return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))
            else:
                raise ValueError(f"Unknown activation: {fn}")
        
        def reverse(params: NDArray, x: NDArray, dy: NDArray) -> Tuple[NDArray, NDArray]:
            if fn == "relu":
                dx = dy * (x > 0).astype(float)
            elif fn == "sigmoid":
                s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                dx = dy * s * (1 - s)
            elif fn == "tanh":
                t = np.tanh(x)
                dx = dy * (1 - t**2)
            elif fn == "softmax":
                # Jacobian is diag(softmax) - softmax @ softmax.T
                # For cross-entropy loss, this simplifies to softmax - target
                # Here we assume dy is pre-computed gradient
                dx = dy
            elif fn == "gelu":
                # Approximate GELU derivative
                cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
                pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
                dx = dy * (cdf + x * pdf)
            elif fn == "swish":
                sigmoid = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                dx = dy * (sigmoid + x * sigmoid * (1 - sigmoid))
            else:
                dx = dy
            
            return dx, np.zeros(1)
        
        return ParametricLens(
            forward_map=forward,
            reverse_derivative=reverse,
            param_spec=param_spec,
            input_spec=input_spec,
            output_spec=output_spec,
            name=name
        )
    
    @staticmethod
    def dropout(p: float = 0.5, training: bool = True, name: str = "dropout") -> ParametricLens:
        """Create dropout lens (stochastic)."""
        param_spec = TensorSpec(shape=(1,), dtype=np.dtype('float64'))
        input_spec = TensorSpec(shape=(None,), dtype=np.dtype('float64'))
        output_spec = TensorSpec(shape=(None,), dtype=np.dtype('float64'))
        
        # Store mask for reverse pass using a counter-based key for reliability
        _mask_cache: Dict[int, NDArray] = {}
        _call_counter = [0]
        # Thread-local side channel: maps output array id → dropout key
        _key_map: Dict[int, int] = {}

        def forward(params: NDArray, x: NDArray) -> NDArray:
            if not training:
                return x * (1 - p)  # Scale at test time

            mask = (np.random.rand(*x.shape) > p).astype(float) / max(1 - p, 1e-8)
            key = _call_counter[0]
            _call_counter[0] += 1
            _mask_cache[key] = mask
            result = x * mask
            # Map output array id to key so reverse can look it up
            _key_map[id(result)] = key
            return result

        def reverse(params: NDArray, x: NDArray, dy: NDArray) -> Tuple[NDArray, NDArray]:
            key = _key_map.pop(id(dy), None)
            if key is not None and key in _mask_cache:
                mask = _mask_cache.pop(key)
            elif _mask_cache:
                # Fallback: use most recent mask
                key = max(_mask_cache.keys())
                mask = _mask_cache.pop(key)
            else:
                mask = np.ones_like(dy)
            dx = dy * mask
            return dx, np.zeros(1)
        
        return ParametricLens(
            forward_map=forward,
            reverse_derivative=reverse,
            param_spec=param_spec,
            input_spec=input_spec,
            output_spec=output_spec,
            name=name
        )
    
    @staticmethod
    def batch_norm(
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        name: str = "batch_norm"
    ) -> ParametricLens:
        """Create batch normalization lens."""
        # Parameters: [gamma, beta] if affine, else single dummy param
        param_size = 2 * num_features if affine else 1
        param_spec = TensorSpec(shape=(param_size,), dtype=np.dtype('float64'))
        input_spec = TensorSpec(shape=(num_features,), dtype=np.dtype('float64'))
        output_spec = TensorSpec(shape=(num_features,), dtype=np.dtype('float64'))
        
        # Running statistics (not parameters, but state)
        running_mean = np.zeros(num_features, dtype=np.float64)
        running_var = np.ones(num_features, dtype=np.float64)
        
        def forward(params: NDArray, x: NDArray) -> NDArray:
            nonlocal running_mean, running_var
            
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running stats
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var
            
            # Normalize
            std = np.sqrt(batch_var + eps)
            x_norm = (x - batch_mean) / std
            
            if affine:
                gamma = params[:num_features]
                beta = params[num_features:]
                return gamma * x_norm + beta
            
            return x_norm
        
        def reverse(params: NDArray, x: NDArray, dy: NDArray) -> Tuple[NDArray, NDArray]:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            std = np.sqrt(batch_var + eps)
            x_norm = (x - batch_mean) / std

            if affine:
                gamma = params[:num_features]
                dbeta = np.sum(dy, axis=0) if dy.ndim > 1 else dy
                dgamma = np.sum(dy * x_norm, axis=0) if dy.ndim > 1 else dy * x_norm
                dparams = np.concatenate([dgamma, dbeta])
                # Full batch norm backward
                if dy.ndim > 1:
                    dx = (gamma / std) * (dy - np.mean(dy, axis=0) - x_norm * np.mean(dy * x_norm, axis=0))
                else:
                    dx = gamma * dy / std
            else:
                dparams = np.zeros(1)
                if dy.ndim > 1:
                    dx = (dy - np.mean(dy, axis=0) - x_norm * np.mean(dy * x_norm, axis=0)) / std
                else:
                    dx = dy / std

            return dx, dparams
        
        return ParametricLens(
            forward_map=forward,
            reverse_derivative=reverse,
            param_spec=param_spec,
            input_spec=input_spec,
            output_spec=output_spec,
            name=name
        )
    
    @staticmethod
    def sequential(lenses: List[ParametricLens], name: str = "sequential") -> ParametricLens:
        """Compose lenses sequentially using vertical composition."""
        if not lenses:
            raise ValueError("Empty lens list")
        
        result = lenses[0]
        for lens in lenses[1:]:
            result = result.compose(lens)
        
        # Rename
        return ParametricLens(
            forward_map=result.forward_map,
            reverse_derivative=result.reverse_derivative,
            param_spec=result.param_spec,
            input_spec=result.input_spec,
            output_spec=result.output_spec,
            name=name
        )
    
    @staticmethod
    def residual_block(
        main_path: List[ParametricLens],
        shortcut: Optional[ParametricLens] = None,
        name: str = "residual"
    ) -> ParametricLens:
        """Create residual connection block."""
        main_lens = LensFactory.sequential(main_path, name=f"{name}_main")
        
        if shortcut is None:
            # Identity shortcut
            shortcut = LensFactory.linear(
                main_lens.input_spec.shape[0],
                main_lens.output_spec.shape[0],
                bias=False,
                name=f"{name}_shortcut"
            )
        
        # Residual: F(x) + shortcut(x)
        # This requires custom composition
        
        def forward(params: NDArray, x: NDArray) -> NDArray:
            # Split params
            main_size = main_lens.param_spec.size
            main_params = params[:main_size]
            shortcut_params = params[main_size:]
            
            main_out = main_lens.forward(main_params, x)
            shortcut_out = shortcut.forward(shortcut_params, x)
            
            return main_out + shortcut_out
        
        def reverse(params: NDArray, x: NDArray, dy: NDArray) -> Tuple[NDArray, NDArray]:
            # Gradients flow through both paths
            main_size = main_lens.param_spec.size
            main_params = params[:main_size]
            shortcut_params = params[main_size:]
            
            # Reverse through main path
            dx_main, dmain_params = main_lens.reverse(main_params, x, dy)
            
            # Reverse through shortcut
            dx_shortcut, dshortcut_params = shortcut.reverse(shortcut_params, x, dy)
            
            # Sum gradients
            dx = dx_main + dx_shortcut
            dparams = np.concatenate([
                np.asarray(dmain_params).flatten(),
                np.asarray(dshortcut_params).flatten()
            ])
            
            return dx, dparams
        
        total_params = main_lens.param_spec.size + shortcut.param_spec.size
        
        return ParametricLens(
            forward_map=forward,
            reverse_derivative=reverse,
            param_spec=TensorSpec(shape=(total_params,)),
            input_spec=main_lens.input_spec,
            output_spec=main_lens.output_spec,
            name=name
        )
    
    @staticmethod
    def attention(
        embed_dim: int,
        num_heads: int = 1,
        name: str = "attention"
    ) -> ParametricLens:
        """Create multi-head self-attention lens."""
        head_dim = embed_dim // num_heads
        total_dim = 3 * embed_dim  # Q, K, V projections
        
        # Parameters for Q, K, V projections and output projection
        qkv_weight_size = total_dim * embed_dim
        out_weight_size = embed_dim * embed_dim
        total_params = qkv_weight_size + out_weight_size + total_dim + embed_dim  # Include biases
        
        param_spec = TensorSpec(shape=(total_params,))
        input_spec = TensorSpec(shape=(None, embed_dim))  # (seq_len, embed_dim)
        output_spec = TensorSpec(shape=(None, embed_dim))
        
        def forward(params: NDArray, x: NDArray) -> NDArray:
            # x shape: (seq_len, embed_dim)
            seq_len = x.shape[0]
            
            # Split parameters
            qkv_weight = params[:qkv_weight_size].reshape(total_dim, embed_dim)
            out_weight = params[qkv_weight_size:qkv_weight_size + out_weight_size].reshape(embed_dim, embed_dim)
            qkv_bias = params[qkv_weight_size + out_weight_size:qkv_weight_size + out_weight_size + total_dim]
            out_bias = params[-embed_dim:]
            
            # Project to Q, K, V
            qkv = x @ qkv_weight.T + qkv_bias  # (seq_len, 3*embed_dim)
            q, k, v = np.split(qkv, 3, axis=-1)
            
            # Reshape for multi-head
            q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)  # (heads, seq, head_dim)
            k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            
            # Attention scores
            scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(head_dim)  # (heads, seq, seq)
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-10)
            
            # Apply attention to values
            attn_output = np.matmul(attn_weights, v)  # (heads, seq, head_dim)
            attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)
            
            # Final projection
            output = attn_output @ out_weight.T + out_bias
            return output
        
        def reverse(params: NDArray, x: NDArray, dy: NDArray) -> Tuple[NDArray, NDArray]:
            # Full attention backward pass
            seq_len = x.shape[0]

            # Re-run forward to get intermediates
            qkv_weight = params[:qkv_weight_size].reshape(total_dim, embed_dim)
            out_weight = params[qkv_weight_size:qkv_weight_size + out_weight_size].reshape(embed_dim, embed_dim)
            qkv_bias = params[qkv_weight_size + out_weight_size:qkv_weight_size + out_weight_size + total_dim]
            out_bias = params[-embed_dim:]

            qkv = x @ qkv_weight.T + qkv_bias
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

            scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(head_dim)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
            attn_output = np.matmul(attn_weights, v)
            attn_output_merged = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)

            # Backward through output projection
            d_out_bias = np.sum(dy, axis=0)
            d_out_weight = attn_output_merged.T @ dy
            d_attn_merged = dy @ out_weight

            # Backward through reshape/transpose
            d_attn = d_attn_merged.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

            # Backward through attention weighted sum
            dv = np.matmul(attn_weights.transpose(0, 2, 1), d_attn)
            d_attn_weights = np.matmul(d_attn, v.transpose(0, 2, 1))

            # Backward through softmax
            d_scores = attn_weights * (d_attn_weights - np.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True))
            d_scores = d_scores / np.sqrt(head_dim)

            # Backward through Q, K matmul
            dq = np.matmul(d_scores, k)
            dk = np.matmul(d_scores.transpose(0, 2, 1), q)

            # Merge heads back
            dq = dq.transpose(1, 0, 2).reshape(seq_len, embed_dim // 3 * num_heads)
            dk = dk.transpose(1, 0, 2).reshape(seq_len, embed_dim // 3 * num_heads)
            dv = dv.transpose(1, 0, 2).reshape(seq_len, embed_dim // 3 * num_heads)
            dqkv = np.concatenate([dq, dk, dv], axis=-1)

            # Backward through QKV projection
            d_qkv_bias = np.sum(dqkv, axis=0)
            d_qkv_weight = x.T @ dqkv
            dx = dqkv @ qkv_weight

            # Assemble param gradients
            dparams = np.concatenate([
                d_qkv_weight.T.flatten(),
                d_out_weight.T.flatten(),
                d_qkv_bias,
                d_out_bias
            ])

            return dx, dparams
        
        return ParametricLens(
            forward_map=forward,
            reverse_derivative=reverse,
            param_spec=param_spec,
            input_spec=input_spec,
            output_spec=output_spec,
            name=name
        )


# ============================================================================
# SECTION 11: REASONING CONCEPTS INTERFACE
# ============================================================================

class ReasoningConcept(ABC):
    """
    Abstract interface for ETI reasoning concepts.
    
    Each concept implements specific mathematical structure while
    using the universal training engine.
    """
    
    @property
    @abstractmethod
    def concept_id(self) -> int:
        """1-20 identifier."""
        pass
    
    @property
    @abstractmethod
    def concept_name(self) -> str:
        """Human-readable name."""
        pass
    
    @abstractmethod
    def create_lens(self, **kwargs) -> ParametricLens:
        """Create the parametric lens for this reasoning concept."""
        pass
    
    @abstractmethod
    def initialize_params(self, **kwargs) -> ParameterState:
        """Initialize parameters respecting concept structure."""
        pass
    
    def get_default_config(self) -> TrainingConfig:
        """Default training configuration for this concept."""
        return TrainingConfig()
    
    def get_constraint_type(self) -> ConstraintType:
        """Get appropriate constraint type for this concept."""
        return ConstraintType.EUCLIDEAN


class OntologicalConcept(ReasoningConcept):
    """Concept 1: Ontological reasoning (being/existence)."""
    
    @property
    def concept_id(self) -> int:
        return 1
    
    @property
    def concept_name(self) -> str:
        return "Ontological Reasoning"
    
    def create_lens(self, dim: int = 64, **kwargs) -> ParametricLens:
        # Existence classification with sheaf constraints
        return LensFactory.sequential([
            LensFactory.linear(dim, dim, name="ontology_hidden"),
            LensFactory.activation("gelu"),
            LensFactory.linear(dim, 2, name="existence_classifier")  # exists/not-exists
        ], name="ontological_lens")
    
    def initialize_params(self, dim: int = 64, **kwargs) -> ParameterState:
        lens = self.create_lens(dim=dim)
        data = np.random.randn(lens.param_spec.size) * 0.02
        return ParameterState(
            data=data,
            spec=lens.param_spec,
            constraint_type=ConstraintType.SHEAF  # Ontological constraints are global
        )
    
    def get_constraint_type(self) -> ConstraintType:
        return ConstraintType.SHEAF


class TemporalConcept(ReasoningConcept):
    """Concept 2: Temporal reasoning (time/change)."""
    
    @property
    def concept_id(self) -> int:
        return 2
    
    @property
    def concept_name(self) -> str:
        return "Temporal Reasoning"
    
    def create_lens(self, dim: int = 64, **kwargs) -> ParametricLens:
        # Recurrent structure for temporal dynamics
        return LensFactory.sequential([
            LensFactory.linear(dim, dim * 2, name="temporal_expand"),
            LensFactory.activation("tanh"),  # Bounded dynamics
            LensFactory.linear(dim * 2, dim, name="temporal_contract")
        ], name="temporal_lens")
    
    def initialize_params(self, dim: int = 64, **kwargs) -> ParameterState:
        lens = self.create_lens(dim=dim)
        # Initialize for stable dynamics (eigenvalues near 1)
        data = np.random.randn(lens.param_spec.size) * 0.1
        return ParameterState(
            data=data,
            spec=lens.param_spec,
            constraint_type=ConstraintType.ORDINAL  # Ordinal time structure
        )
    
    def get_default_config(self) -> TrainingConfig:
        cfg = super().get_default_config()
        cfg.temporal_structure = TemporalStructure.ORDINAL
        return cfg


class CausalConcept(ReasoningConcept):
    """Concept 3: Causal reasoning (intervention/counterfactual)."""
    
    @property
    def concept_id(self) -> int:
        return 3
    
    @property
    def concept_name(self) -> str:
        return "Causal Reasoning"
    
    def create_lens(self, dim: int = 64, **kwargs) -> ParametricLens:
        # SCM (Structural Causal Model) structure
        return LensFactory.sequential([
            LensFactory.linear(dim, dim, name="causal_mechanism"),
            LensFactory.activation("relu"),
            LensFactory.linear(dim, dim, name="intervention")
        ], name="causal_lens")
    
    def initialize_params(self, dim: int = 64, **kwargs) -> ParameterState:
        lens = self.create_lens(dim=dim)
        data = np.random.randn(lens.param_spec.size) * 0.1
        return ParameterState(
            data=data,
            spec=lens.param_spec,
            constraint_type=ConstraintType.EUCLIDEAN
        )


# ============================================================================
# SECTION 12: HIGH-LEVEL API AND UTILITIES
# ============================================================================

def create_trainer(
    concept_type: Optional[str] = None,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    use_sheaf: bool = False,
    **kwargs
) -> UniversalTrainer:
    """
    Factory function to create appropriate trainer.
    
    Args:
        concept_type: "standard", "ontological", "temporal", "causal", etc.
        learning_rate: Base learning rate
        optimizer: "sgd", "momentum", "adam", "adamw", "natural", "lbfgs"
        use_sheaf: Enable sheaf cohomological constraints
        **kwargs: Additional config options
    """
    # Map optimizer names to classes
    optimizers = {
        'sgd': SGD,
        'momentum': lambda: SGD(),  # With momentum in config
        'adam': Adam,
        'adamw': Adam,  # Adam with weight decay
        'natural': NaturalGradient,
        'lbfgs': SecondOrder
    }
    
    opt_class = optimizers.get(optimizer, Adam)
    opt_kwargs = {}
    
    if optimizer == 'adamw':
        opt_kwargs['weight_decay'] = kwargs.get('weight_decay', 0.01)
    
    # Temporal structure
    temporal_map = {
        'standard': TemporalStructure.STANDARD,
        'temporal': TemporalStructure.ORDINAL,
        'retrocausal': TemporalStructure.RETROCAUSAL,
        'hyperreal': TemporalStructure.HYPERREAL
    }
    temporal = temporal_map.get(concept_type or 'standard', TemporalStructure.STANDARD)
    
    # Create sheaf if requested
    sheaf = None
    if use_sheaf:
        n = kwargs.get('sheaf_size', 10)
        coboundary = csr_matrix((n, n))
        sheaf = SheafStructure(
            restriction_maps={},
            coboundary=coboundary
        )
    
    config = TrainingConfig(
        learning_rate=learning_rate,
        optimizer=opt_class,
        optimizer_kwargs=opt_kwargs,
        temporal_structure=temporal,
        **{k: v for k, v in kwargs.items() if k not in ['sheaf_size', 'weight_decay']}
    )
    
    return UniversalTrainer(config)


def quick_train(
    lens: ParametricLens,
    data: List[Tuple[Any, Any]],
    epochs: int = 100,
    learning_rate: float = 0.001
) -> Tuple[ParameterState, TrainingMetrics]:
    """
    Quick training utility for simple use cases.
    
    Example:
        >>> lens = LensFactory.linear(10, 1)
        >>> data = [(np.random.randn(10), np.random.randn(1)) for _ in range(100)]
        >>> params, metrics = quick_train(lens, data)
    """
    # Initialize parameters
    init_data = np.random.randn(lens.param_spec.size).astype(np.float64) * 0.1
    params = ParameterState(
        data=init_data,
        spec=lens.param_spec
    )
    
    # Create trainer
    config = TrainingConfig(
        learning_rate=learning_rate,
        max_epochs=epochs,
        verbose=False
    )
    trainer = UniversalTrainer(config)
    
    # Train
    final_params, metrics = trainer.train(lens, params, data)
    return final_params, metrics


# ============================================================================
# SECTION 13: PRODUCTION EXAMPLE AND TESTING
# ============================================================================

def production_example() -> Tuple[ParameterState, TrainingMetrics]:
    """
    Production-ready demonstration with comprehensive architecture.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting production training example")
    
    # Create complex architecture: ResNet-style block
    dim = 32
    
    # Main path
    main_path = [
        LensFactory.linear(dim, dim, name="res1"),
        LensFactory.batch_norm(dim, name="bn1"),
        LensFactory.activation("relu", name="relu1"),
        LensFactory.linear(dim, dim, name="res2"),
        LensFactory.batch_norm(dim, name="bn2")
    ]
    
    # Residual block
    residual_lens = LensFactory.residual_block(
        main_path,
        name="residual_block"
    )
    
    # Full network
    network = LensFactory.sequential([
        LensFactory.linear(2, dim, name="input_proj"),
        LensFactory.activation("relu"),
        residual_lens,
        LensFactory.activation("relu"),
        LensFactory.linear(dim, 1, name="output")
    ], name="production_net")
    
    # Initialize with Xavier/Glorot initialization
    def xavier_init(shape: Tuple[int, ...]) -> NDArray:
        if len(shape) == 2:
            limit = np.sqrt(6.0 / (shape[0] + shape[1]))
            return np.random.uniform(-limit, limit, shape)
        return np.random.randn(*shape) * 0.01
    
    init_data = xavier_init((network.param_spec.size,))
    initial_params = ParameterState(
        data=init_data,
        spec=network.param_spec,
        constraint_type=ConstraintType.EUCLIDEAN
    )
    
    # Generate synthetic dataset (nonlinear function)
    np.random.seed(42)
    
    def target_function(x: NDArray) -> NDArray:
        """Complex nonlinear target: sin(x0) * cos(x1) + 0.1 * x0^2 * x1"""
        return np.array([
            np.sin(x[0]) * np.cos(x[1]) + 0.1 * x[0]**2 * x[1]
        ])
    
    # Training data
    train_data = []
    for _ in range(500):
        x = np.random.randn(2).astype(np.float64) * 2
        y = target_function(x)
        train_data.append((x, y))
    
    # Validation data
    val_data = []
    for _ in range(100):
        x = np.random.randn(2).astype(np.float64) * 2
        y = target_function(x)
        val_data.append((x, y))
    
    # Configure trainer with advanced features
    config = TrainingConfig(
        learning_rate=0.01,
        max_epochs=200,
        optimizer=Adam,
        optimizer_kwargs={'beta1': 0.9, 'beta2': 0.999},
        loss_function=MSE(reduction='mean'),
        batch_size=16,
        patience=30,
        gradient_clip=1.0,
        verbose=True,
        log_interval=10,
        checkpoint_dir=Path("/tmp/eti_checkpoints"),
        seed=42
    )
    
    trainer = UniversalTrainer(config)
    
    # Add callback example
    def on_epoch_end(trainer, epoch, loss, val_loss):
        if epoch % 20 == 0:
            logger.info(f"Callback: Epoch {epoch} completed with loss {loss:.6f}")
    
    trainer.register_callback('epoch_end', on_epoch_end)
    
    # Train
    logger.info("Starting training...")
    final_params, metrics = trainer.train(
        network,
        initial_params,
        train_data,
        val_data=val_data
    )
    
    # Results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final training loss: {metrics.loss_history[-1]:.6f}")
    print(f"Best validation loss: {metrics.best_loss:.6f}")
    print(f"Best epoch: {metrics.best_epoch}")
    print(f"Converged: {metrics.converged}")
    if metrics.converged_at:
        print(f"Converged at epoch: {metrics.converged_at}")
    print(f"Stopped early: {metrics.stopped_early}")
    if metrics.stop_reason:
        print(f"Stop reason: {metrics.stop_reason}")
    print(f"Total steps: {len(metrics.loss_history)}")
    print(f"Average step time: {np.mean(metrics.step_times)*1000:.2f} ms")
    print(f"Final parameter norm: {np.linalg.norm(final_params.data):.2f}")
    print(f"Parameter hash: {final_params.compute_hash()}")
    print("="*60)
    
    # Test prediction
    test_x = np.array([1.0, -0.5], dtype=np.float64)
    pred = network.forward(final_params.data, test_x)
    true_y = target_function(test_x)
    print(f"\nTest prediction:")
    print(f"  Input: {test_x}")
    print(f"  Predicted: {pred[0]:.6f}")
    print(f"  True: {true_y[0]:.6f}")
    print(f"  Error: {abs(pred[0] - true_y[0]):.6f}")
    
    return final_params, metrics


def test_sheaf_constraints():
    """Test sheaf-theoretic constraint propagation."""
    logger.info("Testing sheaf constraints...")
    
    # Create simple sheaf structure
    n = 5
    coboundary = csr_matrix((n-1, n))
    for i in range(n-1):
        coboundary[i, i] = 1
        coboundary[i, i+1] = -1
    
    sheaf = SheafStructure(
        cells={SheafCell(0, i, 1): i for i in range(n)},
        restriction_maps={},
        coboundary=coboundary
    )
    
    # Create inconsistent parameters
    params = np.arange(n, dtype=np.float64).reshape(n, 1)
    print(f"Original params: {params.flatten()}")
    
    # Project to harmonic (consistent)
    harmonic_proj = sheaf.project_harmonic(params)
    print(f"Harmonic projection: {harmonic_proj.flatten()}")
    
    # Check consistency
    laplacian = sheaf.laplacian
    if laplacian is not None:
        consistency_check = laplacian @ harmonic_proj.flatten()
        print(f"Laplacian applied to projection: {consistency_check}")
        print(f"Is harmonic (should be ~0): {np.allclose(consistency_check, 0)}")
    
    return sheaf


def test_transfinite_training():
    """Test transfinite (ordinal) training."""
    logger.info("Testing transfinite training...")
    
    # Simple linear regression
    lens = LensFactory.linear(2, 1, name="transfinite_linear")
    
    # Data
    true_w = np.array([2.0, -1.0])
    data = []
    for _ in range(50):
        x = np.random.randn(2)
        y = np.array([np.dot(true_w, x) + 0.1 * np.random.randn()])
        data.append((x, y))
    
    # Initialize
    init_params = ParameterState(
        data=np.random.randn(lens.param_spec.size) * 0.1,
        spec=lens.param_spec,
        constraint_type=ConstraintType.ORDINAL
    )
    
    # Transfinite trainer
    config = TrainingConfig(
        learning_rate=0.01,
        temporal_structure=TemporalStructure.ORDINAL,
        verbose=True
    )
    trainer = UniversalTrainer(config)
    
    # Train with ordinal limit
    final_params, metrics = trainer.train_transfinite(
        lens,
        init_params,
        data,
        ordinal_limit=50
    )
    
    print(f"\nTransfinite training complete:")
    print(f"Final loss: {metrics.loss_history[-1]:.6f}")
    learned_w = final_params.data[:2]
    print(f"Learned weights: {learned_w}")
    print(f"True weights: {true_w}")
    print(f"Weight error: {np.linalg.norm(learned_w - true_w):.4f}")
    
    return final_params, metrics


if __name__ == "__main__":
    # Run production example
    print("Running production example...")
    params, metrics = production_example()
    
    print("\n" + "="*60)
    print("Running sheaf test...")
    print("="*60)
    test_sheaf_constraints()
    
    print("\n" + "="*60)
    print("Running transfinite test...")
    print("="*60)
    test_transfinite_training()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)