# Curriculum Engine

## Overview

The curriculum engine orchestrates structured learning through domains, lessons, and rules. It bridges symbolic knowledge representation with neural learning, enabling systematic cognitive development.

## Core Concepts

### Domain

A domain represents a subject area (e.g., "Mathematics", "English", "Physics"):

```python
engine.new_domain(
    domain_id='math',
    name='Mathematics',
    description='Number theory and arithmetic operations'
)
```

### Lesson

A lesson is a teachable unit within a domain:

```python
engine.new_lesson(
    lesson_id='counting_1_10',
    name='Counting 1 to 10',
    domain='math',
    rule_ids=['successor_rule'],
    epochs=100,
    competency_threshold=0.85,
    prerequisite_ids=['number_is_object']
)
```

### Rule

Rules encode symbolic knowledge as patterns:

```python
engine.define_rule(
    rule_id='successor_rule',
    name='Successor Rule',
    domain='math',
    pattern='[n] -> [n+1]',
    weight=2.0
)
```

## Symbolic Encoder

The encoder converts symbols to fixed-length vectors:

```python
from cogniarch.curriculum import SymbolEncoder

encoder = SymbolEncoder(dim=64)

# Base vector (deterministic from hash)
base = encoder.base_vec('OBJECT')

# Learned offset (updated during training)
encoder.update_learned('OBJECT', delta_vector)

# Final encoding (what the model sees)
vec = encoder.encode('OBJECT')

# Sequence encoding
seq_vec = encoder.encode_sequence(['cat', 'is', 'an', 'OBJECT'])
```

### Properties

- **Deterministic**: Same symbol always produces same base vector
- **Invertible**: Can recover symbol from known vocabulary
- **Compositional**: Sequences encoded with positional awareness
- **Trainable**: Learned offsets adapt during training

## Pattern Matcher

Matches inputs to stored rules:

```python
from cogniarch.curriculum import PatternMatcher

matcher = PatternMatcher(encoder)

# Register rules
matcher.add_rule(rule_id='object_identity', pattern='[any] -> OBJECT')
matcher.add_rule(rule_id='successor', pattern='[n] -> [n+1]')

# Match input
matches = matcher.match(input_vector, threshold=0.8)
for match in matches:
    print(f"Rule: {match.rule_id}, Confidence: {match.confidence}")
```

## Rule Memory

Persistent storage for learned rules:

```python
from cogniarch.curriculum import RuleMemory

memory = RuleMemory(memory_dir='cogniarch_memory')

# Store rule
memory.store({
    'rule_id': 'object_identity',
    'pattern': '[any] -> OBJECT',
    'weight': 2.0,
    'confidence': 0.95
})

# Load all rules
rules = memory.load_all()

# Query by domain
math_rules = memory.query(domain='math')
```

## Knowledge Transfer Graph

Tracks transfer between domains:

```python
engine.add_transfer(
    source_domain='english',
    target_domain='math',
    concept='object_identity',
    strength=0.8
)

# Check available transfers
transfers = engine.get_transfers_for('math')
```

## Complete Workflow

```python
from cogniarch import CognitiveSystem, Curriculum

# Initialize
system = CognitiveSystem(
    input_dim=64,
    hidden_dims=[256, 256],
    cl_strategy='ewc_replay'
)

# Define curriculum
curriculum = Curriculum()

# English domain - foundation
curriculum.add_domain('english', 'English', 'Object abstraction')
curriculum.add_rule(
    rule_id='object_identity',
    pattern='[any] -> OBJECT',
    weight=2.0
)
curriculum.add_lesson(
    lesson_id='eng_objects',
    name='Object Identity',
    domain='english',
    rule_ids=['object_identity'],
    epochs=100,
    competency_threshold=0.95
)

# Math domain - builds on English
curriculum.add_domain('math', 'Mathematics', 'Numbers and counting')
curriculum.add_rule(
    rule_id='number_is_object',
    pattern='[number] -> OBJECT',
    weight=1.5
)
curriculum.add_rule(
    rule_id='successor_rule',
    pattern='[n] -> [n+1]',
    weight=2.0
)
curriculum.add_lesson(
    lesson_id='math_numbers',
    name='Numbers as Objects',
    domain='math',
    rule_ids=['number_is_object'],
    prerequisite_ids=['eng_objects'],
    epochs=50
)
curriculum.add_lesson(
    lesson_id='math_counting',
    name='Counting 1-10',
    domain='math',
    rule_ids=['successor_rule'],
    prerequisite_ids=['math_numbers'],
    epochs=100
)

# Follow curriculum
system.follow(curriculum)

# Teach lessons in order
system.teach('eng_objects', object_training_data)
system.teach('math_numbers', number_data)
system.teach('math_counting', counting_data)

# Verify learning
accuracy = system.evaluate('math_counting', test_data)
print(f"Counting accuracy: {accuracy:.3f}")

# Check retention of previous lessons
retention = system.evaluate('eng_objects', object_test_data)
print(f"Object identity retention: {retention:.3f}")
```

## Data Generation

Custom data generators for lessons:

```python
def counting_generator():
    """Generate successor pairs."""
    data = []
    for i in range(1, 10):
        input_vec = encoder.encode(str(i))
        target_vec = encoder.encode(str(i + 1))
        data.append((input_vec, target_vec))
    return data * 20  # Multiple epochs worth

engine.generate_lesson_data(
    lesson_id='counting_1_10',
    custom_generator=counting_generator
)
```

## Competency Gating

Advance only when threshold met:

```python
result = system.teach('counting_1_10', data)

if result.passed_competency:
    print(f"Competency achieved: {result.accuracy:.3f}")
    # Unlock next lesson
    system.unlock('counting_10_99')
else:
    print(f"Retraining needed. Accuracy: {result.accuracy:.3f}")
    # Retry with more epochs
    system reteach('counting_1_10', additional_epochs=50)
```

## Developmental Stages

### Stage 0: Object Permanence

```python
curriculum.add_lesson(
    lesson_id='stage0_objects',
    name='Object Identity',
    rule_pattern='[any] -> OBJECT',
    competency_threshold=0.95
)
```

### Stage 1: Symbol Grounding

```python
curriculum.add_lesson(
    lesson_id='stage1_symbols',
    name='Symbol Recognition',
    rule_pattern='[symbol] -> MEANING',
    prerequisite_ids=['stage0_objects']
)
```

### Stage 2: Sequential Reasoning

```python
curriculum.add_lesson(
    lesson_id='stage2_sequences',
    name='Counting',
    rule_pattern='[n] -> [n+1]',
    prerequisite_ids=['stage1_symbols']
)
```

### Stage 3: Relational Abstraction

```python
curriculum.add_lesson(
    lesson_id='stage3_relations',
    name='Relations',
    rule_pattern='[a] [relation] [b] -> TRUTH_VALUE',
    prerequisite_ids=['stage2_sequences']
)
```

### Stage 4: Multi-hop Inference

```python
curriculum.add_lesson(
    lesson_id='stage4_inference',
    name='Chain Reasoning',
    rule_pattern='[a]->[b], [b]->[c] |- [a]->[c]',
    prerequisite_ids=['stage3_relations']
)
```

### Stage 5: Meta-cognition

```python
curriculum.add_lesson(
    lesson_id='stage5_meta',
    name='Self-Reflection',
    rule_pattern='[belief] -> CONFIDENCE_ASSESSMENT',
    prerequisite_ids=['stage4_inference']
)
```

## Implementation Details

### Encoding Algorithm

```python
def _make_base(symbol: str, dim: int) -> NDArray:
    """Create deterministic base vector from symbol hash."""
    h = hashlib.sha256(symbol.encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-10)
```

### Pattern Matching

```python
def match(self, input_vec: NDArray, threshold: float) -> List[Match]:
    """Find rules matching input."""
    matches = []
    for rule in self.rules:
        similarity = cosine_similarity(input_vec, rule.prototype)
        if similarity > threshold:
            matches.append(Match(
                rule_id=rule.id,
                confidence=similarity * rule.weight
            ))
    return sorted(matches, key=lambda m: m.confidence, reverse=True)
```

## Best Practices

1. **Start Simple**: Begin with object identity before complex reasoning
2. **Prerequisites Matter**: Ensure foundational concepts are mastered first
3. **Monitor Retention**: Regularly evaluate previous lessons
4. **Adjust Thresholds**: Competency thresholds should increase with difficulty
5. **Use Transfer**: Leverage knowledge from related domains

## API Reference

See individual module documentation for complete API.
