# CogniArch Architecture

## Vision

CogniArch is a next-generation cognitive architecture that unifies neural learning with symbolic reasoning through a principled mathematical foundation. Unlike conventional deep learning systems that rely on superficial pattern matching, CogniArch implements genuine cognitive development through:

1. **Categorical Foundation**: Built on Cartesian Reverse Differential Categories (CRDCs) for mathematically rigorous gradient-based learning
2. **Sheaf-Theoretic Knowledge Integration**: Uses sheaf cohomology to ensure global consistency of learned representations
3. **Continual Learning Without Forgetting**: Implements multiple anti-forgetting strategies grounded in biological plausibility
4. **Symbolic-Neural Bridge**: Encodes symbolic rules as differentiable structures trainable via gradient descent
5. **Curriculum-Based Development**: Follows developmental stages from basic abstraction to complex reasoning

## Core Principles

### 1. Mathematical Rigor Over Heuristics

Every component is derived from first principles in category theory, differential geometry, and algebraic topology—not ad-hoc engineering tricks.

### 2. Transparent Representations

All internal states are interpretable through:
- Tangent bundle visualization for understanding learning trajectories
- Sheaf sections for tracking constraint propagation
- Rule memory inspection for symbolic knowledge verification

### 3. Lifelong Learning

The system accumulates knowledge indefinitely without catastrophic forgetting through:
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Gradient Episodic Memory (GEM)
- Progressive Neural Networks (PNN)
- PackNet pruning strategies

### 4. Developmental Stages

Learning follows a curriculum mirroring cognitive development:
- **Stage 0**: Object permanence and identity
- **Stage 1**: Symbol grounding and representation
- **Stage 2**: Sequential reasoning and counting
- **Stage 3**: Relational abstraction
- **Stage 4**: Multi-hop inference
- **Stage 5**: Meta-cognitive reflection

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    COGNITIVE LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Rule Memory │  │Pattern Match│  │ Knowledge Transfer  │ │
│  │ (Symbolic)  │  │  Engine     │  │      Graph          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                  CONTINUAL LEARNING LAYER                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │   EWC    │ │    SI    │ │   GEM    │ │  PNN/PackNet │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
│                                                             │
│  • Fisher Information Matrices  • Importance Tracking       │
│  • Replay Buffers              • Task Embeddings            │
│  • Backward/Forward Transfer Metrics                        │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                 CATEGORICAL FOUNDATION (GRAD)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Parametric Lenses: Para(Lens(C))             │   │
│  │  forward: P × A → B        reverse: P×A×B' → P'×A'   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  • Tangent/Cotangent Bundles  • Sheaf Constraints           │
│  • Multiple Constraint Types  • Enriched Categories         │
│  • Functorial Optimization    • Universal Trainer           │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                    SYMBOLIC ENCODER                         │
│  • Deterministic hash-based base vectors                    │
│  • Learned offset vectors (trainable)                       │
│  • Sequence encoding with positional awareness              │
│  • Invertible for known symbols                             │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
cogniarch/
├── src/
│   ├── core/           # GRAD: Categorical foundation
│   │   ├── lens.py     # Parametric lenses
│   │   ├── tangent.py  # Differential geometry
│   │   ├── sheaf.py    # Sheaf theory
│   │   ├── trainer.py  # Universal trainer
│   │   └── constraints.py
│   ├── continual/      # Continual learning strategies
│   │   ├── ewc.py
│   │   ├── si.py
│   │   ├── gem.py
│   │   ├── pnn.py
│   │   └── metrics.py
│   ├── curriculum/     # Curriculum engine
│   │   ├── encoder.py  # Symbolic encoder
│   │   ├── rules.py    # Rule memory
│   │   ├── matcher.py  # Pattern matching
│   │   └── engine.py   # Main orchestrator
│   └── utils/          # Utilities
│       ├── logging.py
│       ├── checkpoint.py
│       └── visualization.py
├── docs/
│   ├── architecture/   # Design documents
│   ├── api/            # API reference
│   └── tutorials/      # Usage guides
├── experiments/        # Experimental scripts
├── benchmarks/         # Performance benchmarks
└── tests/              # Test suite
```

## Key Innovations

### 1. Parametric Lenses

Unlike standard neural networks, every component is a parametric lens obeying categorical laws:
- **Get**: Extract current parameters
- **Put**: Update parameters given loss gradient
- **Reverse**: Compute gradients respecting chain rule

This enables:
- Compositional architectures with guaranteed gradient flow
- Modular components that can be rearranged without retraining
- Formal verification of learning properties

### 2. Sheaf-Theoretic Constraint Propagation

Knowledge constraints propagate through the network as sheaf sections:
- Local consistency enforced at each layer
- Global consistency via sheaf cohomology
- Automatic detection of contradictory beliefs

### 3. Hybrid Symbolic-Neural Learning

Rules are not separate from neural weights:
- Rules have learnable confidence scores
- Neural activations can trigger rule application
- Rules constrain gradient updates
- Bidirectional transfer between symbolic and subsymbolic

## Getting Started

```python
from cogniarch import CognitiveSystem, Curriculum

# Initialize system
system = CognitiveSystem(
    input_dim=64,
    hidden_dims=[256, 256],
    output_dim=64,
    cl_strategy='ewc_replay'
)

# Define curriculum
curriculum = Curriculum()
curriculum.add_domain('objects', 'Object Identity')
curriculum.add_lesson(
    domain='objects',
    lesson_id='identity',
    rule_pattern='[any] -> OBJECT',
    epochs=100
)

# Train
system.follow(curriculum)
system.teach('identity', training_data)

# Query
result = system.query('What is a cat?')
print(result)  # "OBJECT"
```

## Mathematical Foundations

### Cartesian Reverse Differential Categories

A CRDC is a category C with:
- Finite products (cartesian structure)
- Reverse derivative operator R[f]: A × B' → A' for each f: A → B
- Satisfying axioms: linearity, chain rule, symmetry

### Parametric Lenses

A parametric lens in category C consists of:
- Parameter object P
- Input object A  
- Output object B
- Morphism f: P × A → B
- Reverse derivative R[f]: (P × A) × B' → P' × A'

### Sheaves on Posets

A presheaf F on poset (P, ≤) assigns:
- Set F(p) to each p ∈ P
- Restriction map F(q) → F(p) for p ≤ q
- Sheaf condition: compatible local data glues uniquely

## Roadmap

- [x] Core categorical foundation
- [x] Continual learning integration
- [x] Curriculum engine
- [ ] Advanced reasoning modules
- [ ] Multi-modal perception
- [ ] Language understanding
- [ ] Meta-learning capabilities
- [ ] Distributed training

## Citation

```bibtex
@article{cogniarch2024,
  title={CogniArch: A Categorical Foundation for Cognitive AI},
  author={ETI Research Division},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.
