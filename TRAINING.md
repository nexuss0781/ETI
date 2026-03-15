# ETI Training Guide
## Extraterrestrial Intelligence — Full System Training Reference

---

## What This System Is

A three-layer learning stack:

```
curriculum_engine.py   ← you define what to teach (domains, rules, lessons)
continual_learning.py  ← ensures the model never forgets old tasks
GRAD.py                ← the neural engine that does the actual learning
```

The model learns like a student:
- You define **rules** (the textbook)
- You define **lessons** (what to practice)
- The model sees the rule + a few examples and **generalises** — no need for millions of samples
- Rules are **permanent** — loaded every session from disk, never forgotten
- Neural weights are saved too — the model picks up where it left off

---

## Quick Start

```python
from curriculum_engine import CurriculumEngine

# Create engine (loads previous state automatically if it exists)
engine = CurriculumEngine.create(
    memory_dir="eti_memory",   # all state saved here
    cl_strategy="ewc",         # anti-forgetting strategy
    hidden_dims=[256, 256],    # network size
    learning_rate=0.003,
    epochs_per_lesson=50,
    verbose=True,
)

# Define a domain
engine.new_domain("math", "Mathematics", "Numbers and logic")

# Define a rule (saved permanently to disk)
engine.define_rule(
    rule_id="count_rule",
    name="Counting Rule",
    domain="math",
    pattern="[object] * N = N objects",
    weight=2.0,   # high weight = stronger gradient signal
)

# Create a lesson
engine.new_lesson(
    lesson_id="math_counting",
    name="Counting",
    domain="math",
    rule_ids=["count_rule"],
    epochs=50,
)

# Generate training data from the rule
engine.generate_lesson_data("math_counting")

# Teach it
engine.teach_lesson("math_counting")

# Save everything
engine.save()
```

---

## Core Concepts

### Rules
Rules are the **permanent memory** of the system. They are stored in `memory/rules.json` and loaded every session.

| Field | Purpose |
|---|---|
| `rule_id` | Unique identifier |
| `name` | Human-readable name |
| `domain` | Which domain this belongs to |
| `pattern` | Symbolic description e.g. `[obj]*N = N objects` |
| `weight` | How strongly this rule biases training (0–2). Default 1.0 |
| `confidence` | Accumulated from examples (0–1). Grows as examples match |

**Weight guide:**
- `2.0` — foundational rule, very important, teach hard
- `1.5` — important supporting rule
- `1.0` — standard rule
- `0.5` — weak hint, low priority

### Lessons
A lesson is one training unit. It references one or more rules and has training data attached.

```python
engine.new_lesson(
    lesson_id="my_lesson",
    name="My Lesson",
    domain="my_domain",
    rule_ids=["rule_1", "rule_2"],
    difficulty=0.5,              # 0=easy, 1=hard (informational)
    competency_threshold=0.75,   # model must hit 75% accuracy to pass
    epochs=50,
    prerequisite_ids=["earlier_lesson"],  # must complete this first
)
```

### Domains
Domains are fully user-defined subject areas. Nothing is hardcoded.

```python
engine.new_domain(
    domain_id="chemistry",
    name="Chemistry",
    description="Elements, compounds, reactions",
    transfer_from=["math", "physics"],  # borrow representations from these
)
```

---

## Built-in Domains (Ready to Use)

Use `create_full_curriculum()` to get all four pre-built domains:

```python
from curriculum_engine import create_full_curriculum

engine = create_full_curriculum(
    memory_dir="eti_memory",
    strategy="ewc",
    hidden_dims=[256, 256],
    learning_rate=0.003,
    epochs_per_lesson=50,
)

engine.teach_all_domains()  # teaches math -> english -> physics -> programming
```

### Domain 1: Math (foundation)
Teaches the model symbolic reasoning from scratch.

| Lesson | What it learns | Rule |
|---|---|---|
| `math_objects` | Everything is an object | `[object] = [object]` |
| `math_counting` | N objects = count N | `[object]*N = N objects` |
| `math_comparison` | A < B, A = B, A > B | comparison rules |
| `math_ordering` | Sort a sequence ascending | ordering rule |

### Domain 2: English (transfers from math)
Builds on symbolic identity from math.

| Lesson | What it learns |
|---|---|
| `eng_vocab` | Each word has a stable embedding |
| `eng_categories` | Classify: noun / verb / adjective |

### Domain 3: Physics (transfers from math)
Learns physical formulas as function mappings.

| Lesson | Formula |
|---|---|
| `phys_fma` | F = m × a |
| `phys_ke` | KE = 0.5 × m × v² |

### Domain 4: Programming (transfers from math + english)
Classifies code tokens and learns assignment.

| Lesson | What it learns |
|---|---|
| `prog_tokens` | keyword / operator / identifier / literal |
| `prog_assignment` | `var = expression` |

---

## Teaching a Single Lesson

```python
# Option 1: auto-generate data from rules
engine.generate_lesson_data("math_counting")
engine.teach_lesson("math_counting")

# Option 2: provide your own data
data = [(input_vector, target_vector), ...]
lesson = engine._find_lesson("math_counting")
lesson.train_data = data
engine.teach_lesson("math_counting")

# Option 3: custom generator function
engine.generate_lesson_data(
    "my_lesson",
    custom_generator=lambda: my_data_fn(),
)
engine.teach_lesson("my_lesson")
```

---

## Teaching a Full Domain

```python
# Teaches all lessons in dependency order, skips already-completed ones
engine.teach_domain("math")

# Force retrain even if already completed
engine.teach_domain("math", skip_completed=False)
```

---

## Presenting Examples (Few-Shot Rule Matching)

After teaching, you can show the model examples. It matches them to the closest rule and strengthens confidence — no retraining needed.

```python
result = engine.present_example(
    tokens=["object", "object", "object"],
    domain="math",
)

print(result["matched_rule"])   # "Counting Rule"
print(result["similarity"])     # 0.82
print(result["confidence"])     # 0.65 (grows with each example)
print(result["explanation"])    # human-readable match description
```

The more examples you present, the higher the rule's `confidence` grows. This is the few-shot generalisation mechanism — the model sees the rule + a handful of examples and locks in understanding.

---

## Querying the Model

```python
import numpy as np

# Query with a raw vector
x = engine.encoder.encode("cat")
result = engine.query(x, domain="english")

print(result["prediction"])    # model output vector
print(result["matched_rule"])  # closest rule by pattern matching
print(result["similarity"])    # cosine similarity to that rule

# Query with any numpy array
result = engine.query(np.array([0.5, 1.2, -0.3, ...]))
```

---

## Cross-Domain Knowledge Transfer

Transfer pre-warms the encoder so the target domain starts with representations already shaped by the source domain's rules.

```python
# Enable transfer before teaching the target domain
engine.enable_transfer("chemistry", ["math", "physics"])

# Now teach chemistry — it starts with math+physics embeddings
engine.teach_domain("chemistry")
```

Transfer chain in the built-in curriculum:
```
math
 ├── english
 ├── physics
 └── programming (also from english)
```

---

## Anti-Forgetting Strategies

Set via `cl_strategy` when creating the engine. The model uses this strategy to protect old knowledge while learning new tasks.

| Strategy | Best for | Notes |
|---|---|---|
| `ewc` | General use | Elastic Weight Consolidation. Good default |
| `online_ewc` | Long curricula | EWC with exponential moving average |
| `si` | Continuous streams | Synaptic Intelligence, online path integral |
| `replay` | Small curricula | Stores and replays old examples |
| `gem` | Strict no-forgetting | Gradient projection, slowest but strongest |
| `pnn` | Isolated tasks | Progressive Neural Networks, grows network |
| `packnet` | Fixed capacity | Prunes and assigns subnetworks per task |
| `lwf` | Distillation | Learning without Forgetting, KL divergence |
| `ewc_replay` | Best accuracy | EWC + replay combined |
| `si_replay` | Best accuracy | SI + replay combined |

**Recommendation:** Start with `ewc` or `ewc_replay`. Use `gem` if you absolutely cannot tolerate any forgetting.

---

## Network Size Guide

Set via `hidden_dims` when creating the engine.

| Task complexity | Recommended `hidden_dims` |
|---|---|
| Simple rules, few lessons | `[64, 64]` |
| Standard curriculum | `[128, 128]` |
| Large curriculum, many domains | `[256, 256]` |
| Complex reasoning | `[512, 256, 128]` |

The input/output dimensions are set automatically from the first lesson's data.

---

## Epochs Guide

| Lesson type | Recommended epochs |
|---|---|
| Identity / anchoring | 20–30 |
| Classification (few classes) | 40–60 |
| Formula mapping | 60–80 |
| Complex ordering / reasoning | 80–100 |

Use `competency_threshold` to auto-retry if accuracy is too low:

```python
engine.new_lesson(
    ...,
    competency_threshold=0.80,  # retrain up to 3x if accuracy < 80%
    epochs=50,
)
```

---

## Saving and Loading

State is saved automatically after each lesson. To save manually:

```python
engine.save()
```

What gets saved to `memory_dir/`:
- `rules.json` — all rules with weights and confidence (permanent)
- `encoder.pkl` — learned symbol embeddings
- `engine_state.json` — domains, lessons, completed list
- `checkpoints/` — neural network weights (via continual_learning)

To reload in a new session — just create the engine with the same `memory_dir`:

```python
engine = CurriculumEngine.create(memory_dir="eti_memory")
# Everything is restored automatically
```

---

## Building a Custom Domain from Scratch

```python
from curriculum_engine import CurriculumEngine

engine = CurriculumEngine.create(memory_dir="eti_memory")

# 1. Define the domain
engine.new_domain("biology", "Biology", "Living systems and classification")

# 2. Define rules (permanent)
engine.define_rule("cell_rule",    "Cell",    "biology", "[cell] = unit of life",       weight=2.0)
engine.define_rule("dna_rule",     "DNA",     "biology", "[DNA] encodes [protein]",     weight=2.0)
engine.define_rule("kingdom_rule", "Kingdom", "biology", "[organism] belongs [kingdom]", weight=1.5)

# 3. Create lessons with prerequisites
engine.new_lesson("bio_cell",    "Cell Basics",    "biology", ["cell_rule"],    epochs=40)
engine.new_lesson("bio_dna",     "DNA Encoding",   "biology", ["dna_rule"],     epochs=60,
                  prerequisite_ids=["bio_cell"])
engine.new_lesson("bio_kingdom", "Classification", "biology", ["kingdom_rule"], epochs=50,
                  prerequisite_ids=["bio_cell"])

# 4. Provide training data
import numpy as np

def my_cell_data():
    enc = engine.encoder
    data = []
    for _ in range(300):
        x = enc.encode("cell") + np.random.randn(64) * 0.05
        y = enc.encode("cell")
        data.append((x, y))
    return data

engine.generate_lesson_data("bio_cell", custom_generator=my_cell_data)

# 5. Teach
engine.teach_domain("biology")

# 6. Present examples
engine.present_example(["cell", "divides", "cell"], domain="biology")

engine.save()
```

---

## Using GRAD.py Directly (Advanced)

For custom training loops that bypass the curriculum engine entirely.

```python
from GRAD import LensFactory, TrainingConfig, UniversalTrainer, MSE

# Build a network
lens = LensFactory.mlp(
    input_dim=64,
    output_dim=10,
    hidden_dims=[256, 256],
    activation="relu",
)

# Configure training
config = TrainingConfig(
    learning_rate=0.003,
    batch_size=32,
    max_epochs=100,
    gradient_clip=1.0,
    verbose=True,
)

# Train
trainer = UniversalTrainer(config)
params, metrics = trainer.train(lens, lens.init_params(), train_data, val_data=val_data)

print(f"Final loss: {metrics.loss_history[-1]:.6f}")
print(f"Converged: {metrics.converged}")
```

### Available Optimizers

```python
from GRAD import Adam, SGD, RMSProp, NaturalGradient, LBFGS

# Set in TrainingConfig
config = TrainingConfig(optimizer="adam")    # adam | sgd | rmsprop | natural | lbfgs
```

### Available Loss Functions

```python
from GRAD import MSE, CrossEntropy, HuberLoss, ContrastiveLoss, FocalLoss
```

### Available Activations

`relu`, `tanh`, `sigmoid`, `gelu`, `swish`, `leaky_relu`, `elu`

---

## Using continual_learning.py Directly (Advanced)

For multi-task training without the curriculum layer.

```python
from continual_learning import ContinualLearningSystem

system = ContinualLearningSystem.create(
    input_dim=64,
    output_dim=10,
    strategy="ewc_replay",
    hidden_dims=[256, 256],
    learning_rate=0.003,
    epochs_per_task=50,
    verbose=True,
)

# Teach tasks sequentially
system.teach("task_1", train_data=data_1, competency_threshold=0.85)
system.teach("task_2", train_data=data_2, competency_threshold=0.85)
system.teach("task_3", train_data=data_3)

# Check forgetting
print(system.forgetting_report())

# Save / load
system.save("my_model.pkl")
system.load("my_model.pkl")
```

---

## Monitoring Training

```python
# Full engine report
print(engine.report())

# Rule memory summary
print(engine.rule_memory.summary())

# CL system forgetting report
if engine._cl_system:
    print(engine._cl_system.forgetting_report())
```

Example output:
```
============================================================
CURRICULUM ENGINE REPORT
============================================================
  Memory dir : eti_memory
  Strategy   : ewc
  Domains    : 2
  Completed  : 3 lessons

RuleMemory (6 rules):
  [english     ] Adjective                      w=1.50  conf=0.72
  [english     ] Noun                           w=1.50  conf=0.81
  [math        ] Counting Rule                  w=2.00  conf=0.95
  [math        ] Object Identity                w=2.00  conf=0.98

  Domain: Mathematics
    [DONE] math_objects               acc=0.923  diff=0.1
    [DONE] math_counting              acc=0.871  diff=0.3
    [TODO] math_comparison            acc=0.000  diff=0.4

  Domain: English
    [DONE] eng_vocab                  acc=0.961  diff=0.2
============================================================
```

---

## Common Issues and Fixes

**Dimension mismatch between lessons**
Different lessons may have different output sizes (e.g. 64-dim identity vs 3-dim classification). The engine pads/truncates automatically, but for best results keep output dims consistent within a domain or use separate CL systems per domain.

**Low accuracy on a lesson**
- Increase `epochs`
- Lower `learning_rate` (try `0.001`)
- Increase `hidden_dims`
- Set `competency_threshold` so it auto-retries

**Model forgetting old tasks**
- Switch from `ewc` to `ewc_replay` or `gem`
- Increase `lambda_ewc` (default 5000, try 10000–50000)
- Increase `replay_capacity` (default 2000)

**Rules not matching examples**
- Lower `match_threshold` in `CurriculumEngine.create()` (default 0.6, try 0.4)
- Make rule patterns more descriptive
- Present more examples to build confidence

**Training is slow**
- Reduce `hidden_dims`
- Reduce `epochs_per_lesson`
- Reduce `n_samples` in `generate_lesson_data()`

---

## File Structure After Training

```
eti_memory/
├── rules.json          ← permanent rule memory (human-readable)
├── encoder.pkl         ← symbol embeddings (binary)
├── engine_state.json   ← domains, lessons, completed list
└── checkpoints/
    └── checkpoint_*.pkl  ← neural network weights per task
```

The `rules.json` file is human-readable and editable. You can manually adjust rule weights between sessions to steer the model's priorities.

---

## Summary: Training Workflow

```
1. Create engine          CurriculumEngine.create(...)
2. Define domain          engine.new_domain(...)
3. Define rules           engine.define_rule(...)   ← permanent
4. Create lessons         engine.new_lesson(...)
5. Generate data          engine.generate_lesson_data(...)
6. Teach                  engine.teach_domain(...)
7. Present examples       engine.present_example(...)  ← few-shot
8. Query                  engine.query(...)
9. Save                   engine.save()
10. Next session          CurriculumEngine.create(same memory_dir)  ← resumes
```
