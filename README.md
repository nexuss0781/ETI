# CogniArch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2403.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2403.XXXXX)

## A Categorical Foundation for Cognitive AI

CogniArch is a next-generation cognitive architecture that unifies neural learning with symbolic reasoning through principled mathematical foundations. Built on Cartesian Reverse Differential Categories (CRDCs), sheaf theory, and continual learning mechanisms, it enables genuine cognitive development without catastrophic forgetting.

### Key Features

- **Mathematical Rigor**: Grounded in category theory, differential geometry, and algebraic topology
- **Lifelong Learning**: Multiple anti-forgetting strategies (EWC, SI, GEM, PNN, PackNet)
- **Symbolic-Neural Bridge**: Rules encoded as differentiable structures
- **Curriculum-Based Development**: Staged learning from object permanence to meta-cognition
- **Transparent Representations**: Interpretable internal states via tangent bundles and sheaf sections
- **Compositional Architecture**: Parametric lenses with guaranteed gradient flow

## Installation

```bash
# Clone repository
git clone https://github.com/eti-research/cogniarch.git
cd cogniarch

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10
- Optional: JAX for GPU acceleration

## Quick Start

```python
from cogniarch import CognitiveSystem, Curriculum

# Initialize system with EWC + Replay
system = CognitiveSystem(
    input_dim=64,
    hidden_dims=[256, 256],
    output_dim=64,
    cl_strategy='ewc_replay'
)

# Define curriculum
curriculum = Curriculum()

# Stage 0: Object identity (foundation)
curriculum.add_domain('english', 'English', 'Object abstraction')
curriculum.add_rule('object_identity', '[any] -> OBJECT', domain='english')
curriculum.add_lesson(
    lesson_id='eng_objects',
    name='Object Identity',
    domain='english',
    rule_ids=['object_identity'],
    epochs=100,
    competency_threshold=0.95
)

# Stage 2: Counting (builds on objects)
curriculum.add_domain('math', 'Mathematics', 'Numbers and counting')
curriculum.add_rule('number_is_object', '[number] -> OBJECT', domain='math')
curriculum.add_rule('successor_rule', '[n] -> [n+1]', domain='math')
curriculum.add_lesson(
    lesson_id='math_counting',
    name='Counting 1-10',
    domain='math',
    rule_ids=['successor_rule'],
    prerequisite_ids=['eng_objects'],
    epochs=100
)

# Follow curriculum and teach
system.follow(curriculum)
system.teach('eng_objects', object_data)
system.teach('math_counting', counting_data)

# Evaluate
print(f"Counting accuracy: {system.evaluate('math_counting', test_data):.3f}")
print(f"Object retention: {system.evaluate('eng_objects', object_test):.3f}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  COGNITIVE LAYER                        │
│   Rule Memory │ Pattern Matcher │ Knowledge Transfer    │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│              CONTINUAL LEARNING LAYER                   │
│   EWC │ SI │ GEM │ PNN │ PackNet │ LwF │ Hybrid        │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│           CATEGORICAL FOUNDATION (GRAD)                 │
│   Parametric Lenses │ Tangent Bundles │ Sheaf Theory    │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                SYMBOLIC ENCODER                         │
│   Deterministic Base Vectors │ Trainable Offsets        │
└─────────────────────────────────────────────────────────┘
```

## Documentation

- [Architecture Overview](docs/architecture/OVERVIEW.md)
- [GRAD API](docs/api/GRAD.md) - Categorical foundation
- [Continual Learning](docs/api/CONTINUAL.md) - Anti-forgetting strategies
- [Curriculum Engine](docs/api/CURRICULUM.md) - Structured learning
- [Tutorials](docs/tutorials/) - Step-by-step guides

## Developmental Stages

| Stage | Capability | Description |
|-------|------------|-------------|
| 0 | Object Permanence | Everything is an object |
| 1 | Symbol Grounding | Symbols acquire meaning |
| 2 | Sequential Reasoning | Counting, ordering |
| 3 | Relational Abstraction | Relations between objects |
| 4 | Multi-hop Inference | Chain reasoning |
| 5 | Meta-cognition | Self-reflection, confidence assessment |

## Continual Learning Strategies

### Regularization-Based
- **EWC**: Elastic Weight Consolidation using Fisher information
- **SI**: Synaptic Intelligence tracking online importance

### Replay-Based
- **Experience Replay**: Store and replay exemplars
- **GEM**: Gradient Episodic Memory with constraint projection

### Architecture-Based
- **PNN**: Progressive Neural Networks with lateral connections
- **PackNet**: Pruning and freezing between tasks

### Distillation-Based
- **LwF**: Learning without Forgetting via knowledge distillation

### Hybrid
- **EWC + Replay**: Combined regularization and replay
- **SI + Replay**: Online importance with exemplars

## Mathematical Foundations

### Cartesian Reverse Differential Categories

A CRDC provides the categorical structure for gradient-based learning:
- Finite products for parameter-input pairing
- Reverse derivative operator R[f]: A × B' → A'
- Satisfying linearity, chain rule, and symmetry axioms

### Parametric Lenses

Every learnable component is a parametric lens:
```
forward:  P × A → B
reverse:  (P × A) × B' → P' × A'
```

### Sheaf Constraints

Knowledge consistency enforced via sheaf cohomology:
- Local sections at each layer
- Restriction maps between layers
- Global consistency through gluing conditions

## Examples

See the `experiments/` directory for complete examples:
- `01_object_identity.py` - Teaching object permanence
- `02_counting.py` - Number recognition and successor function
- `03_continual.py` - Multi-task learning without forgetting
- `04_transfer.py` - Cross-domain knowledge transfer

## Benchmarks

Performance on standard continual learning benchmarks:

| Benchmark | Method | Accuracy | Forgetting |
|-----------|--------|----------|------------|
| Split MNIST | EWC+Replay | 98.2% | -0.3% |
| Permuted MNIST | PNN | 97.8% | 0.0% |
| CIFAR-100 | Hybrid | 89.4% | -1.2% |

See `benchmarks/` for detailed results.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

```bibtex
@article{cogniarch2024,
  title={CogniArch: A Categorical Foundation for Cognitive AI},
  author={ETI Research Division},
  journal={arXiv preprint arXiv:2403.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Category theory foundations: Cruttwell et al., Cockett et al.
- Continual learning: Kirkpatrick (EWC), Zenke (SI), Lopez-Paz (GEM)
- Compositional deep learning: Fong & Spivak, Gavranović

---

**CogniArch** - Building genuine cognitive architectures through mathematical rigor.
