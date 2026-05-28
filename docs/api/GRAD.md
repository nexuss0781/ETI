# GRAD: Categorical Foundation

## Overview

GRAD (Gradient Reverse Derivative) provides the mathematical foundation for CogniArch using Cartesian Reverse Differential Categories (CRDCs). This module implements parametric lenses, tangent bundles, and universal training with rigorous categorical semantics.

## Mathematical Background

### Cartesian Reverse Differential Categories

A **Cartesian Reverse Differential Category** is a category C equipped with:

1. **Finite Products**: Terminal object 1 and binary products A × B
2. **Reverse Derivative**: For each morphism f: A → B, a morphism R[f]: A × B' → A'
3. **Axioms**:
   - Linearity: R[f](x, y₁ + y₂) = R[f](x, y₁) + R[f](x, y₂)
   - Chain Rule: R[g ∘ f] = R[f] ∘ (id × R[g])
   - Symmetry: R[R[f]] satisfies symmetry conditions
   - Compatibility with cartesian structure

### Parametric Lenses

A **parametric lens** captures the structure of learnable components:

```
Para(Lens(C)) consists of:
- Parameter object P
- Input object A
- Output object B
- Forward: f: P × A → B
- Reverse: R[f]: (P × A) × B' → P' × A'
```

The reverse derivative computes both input gradients and parameter updates simultaneously.

## Core Components

### TangentBundle

Represents tangent vectors in the differential structure:

```python
@dataclass
class TangentBundle:
    base_point: NDArray    # Position in manifold
    fiber_vector: NDArray  # Direction/velocity
    
    def norm(self) -> float: ...
    def __add__(self, other) -> TangentBundle: ...
    def inner_product(self, other) -> float: ...
```

### CotangentBundle

Represents covectors (gradients live here):

```python
@dataclass
class CotangentBundle:
    base_point: NDArray
    covector: NDArray
    
    def apply_to(self, tangent: TangentBundle) -> float: ...
    def musical_isomorphism(self, metric=None) -> TangentBundle: ...
```

### ParametricLens

Abstract base for all learnable components:

```python
class ParametricLens(ABC):
    @abstractmethod
    def forward(self, params: P, input_data: A) -> B: ...
    
    @abstractmethod
    def reverse(self, params: P, input_data: A, output_tangent: B) -> Tuple[A, P]: ...
```

### UniversalTrainer

Implements functorial gradient descent:

```python
class UniversalTrainer:
    def train(self, lens: ParametricLens, data: Dataset, config: TrainingConfig) -> TrainingMetrics: ...
```

## Constraint Types

Supports multiple geometric structures:

```python
class ConstraintType(IntEnum):
    EUCLIDEAN = 0      # Standard vector space
    RIEMANNIAN = 1     # Manifold with metric
    DISCRETE = 2       # Categorical/Gumbel-softmax
    ORDINAL = 3        # Transfinite ordinals
    SHEAF = 4          # Cohomological constraints
    HYPERREAL = 5      # Non-standard analysis
    QUANTUM = 6        # Density matrices
```

## Usage Example

```python
from cogniarch.core import (
    ParametricLens, UniversalTrainer, 
    TrainingConfig, MSE, Adam
)

# Define a simple lens
class LinearLens(ParametricLens):
    def __init__(self, input_dim, output_dim):
        self.spec = TensorSpec(shape=(output_dim, input_dim))
    
    def forward(self, params, x):
        W = params.reshape(self.spec.shape)
        return W @ x
    
    def reverse(self, params, x, output_tangent):
        W = params.reshape(self.spec.shape)
        grad_x = W.T @ output_tangent
        grad_W = np.outer(output_tangent, x)
        return grad_x, grad_W.flatten()

# Train
lens = LinearLens(64, 10)
trainer = UniversalTrainer(
    optimizer='adam',
    loss=MSE(),
    config=TrainingConfig(learning_rate=0.01, epochs=100)
)

metrics = trainer.train(lens, training_data)
print(f"Final loss: {metrics.final_loss}")
```

## Implementation Details

### Lens Composition

Lenses compose via categorical composition:

```python
def compose(f: ParametricLens, g: ParametricLens) -> ParametricLens:
    """Create sequential composition g ∘ f."""
    return ComposedLens(f, g)
```

The reverse derivative automatically applies the chain rule.

### Sheaf Constraints

Sheaf-theoretic constraints enforce global consistency:

```python
@dataclass
class SheafConstraint:
    cells: List[SheafCell]
    restriction_maps: Dict[Tuple[int, int], NDArray]
    
    def check_consistency(self, sections: Dict[int, NDArray]) -> bool: ...
    def project_to_consistent(self, sections) -> Dict[int, NDArray]: ...
```

### Optimization

Multiple optimizers supported:

- **SGD**: Stochastic gradient descent with momentum
- **Adam**: Adaptive moment estimation
- **RiemannianAdam**: Adam on manifolds
- **NaturalGradient**: Second-order optimization

## API Reference

See individual module documentation for complete API reference.

## References

1. Cruttwell, W., et al. "Deep Learning with Parametric Lenses." arXiv:2403.XXXXX (2024)
2. Cockett, J.R.B., et al. "Reverse Derivative Categories." CSL 2020
3. Fong, B., Spivak, D.I. "Seven Sketches in Compositionality." Cambridge University Press (2018)
