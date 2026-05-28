# Continual Learning Module

## Overview

The continual learning module implements strategies for lifelong learning without catastrophic forgetting. It integrates seamlessly with the GRAD categorical foundation and provides multiple biologically-inspired mechanisms.

## Strategies

### 1. Regularization-Based

#### Elastic Weight Consolidation (EWC)

Penalizes changes to parameters important for previous tasks:

```
L(θ) = L_task(θ) + λ/2 * Σ_i F_i (θ_i - θ*_i)²
```

Where F_i is the Fisher information diagonal.

```python
from cogniarch.continual import EWC

ewc = EWC(lambda_=0.5)
ewc.consolidate(task_id=0, params=optimal_params, fisher=fisher_info)
loss = ewc.regularized_loss(current_params, task_loss)
```

#### Synaptic Intelligence (SI)

Tracks parameter importance online during training:

```python
from cogniarch.continual import SynapticIntelligence

si = SynapticIntelligence(xi=0.1)
# During training, SI accumulates importance weights
si.update_importance(params, gradients)
```

### 2. Replay-Based

#### Experience Replay

Stores exemplars from previous tasks:

```python
from cogniarch.continual import ReplayBuffer

buffer = ReplayBuffer(capacity=1000)
buffer.add(task_id=0, samples=training_data)
replay_samples = buffer.sample(batch_size=32)
```

#### Gradient Episodic Memory (GEM)

Constrains gradients to not increase loss on previous tasks:

```python
from cogniarch.continual import GEM

gem = GEM(buffer_size=500)
gem.store_task_gradients(task_id, gradients)
projected_grad = gem.project_gradient(current_grad)
```

### 3. Architecture-Based

#### Progressive Neural Networks (PNN)

Adds new columns for each task with lateral connections:

```python
from cogniarch.continual import PNN

pnn = PNN(base_architecture=[256, 256])
pnn.add_column(task_id=0)  # First task
pnn.add_column(task_id=1)  # New task with connections to column 0
```

#### PackNet

Prunes unimportant weights between tasks:

```python
from cogniarch.continual import PackNet

packnet = PackNet(pruning_fraction=0.75)
packnet.prune_after_training(task_id)
# Frozen weights remain fixed for subsequent tasks
```

### 4. Distillation-Based

#### Learning without Forgetting (LwF)

Uses old model outputs as soft targets:

```python
from cogniarch.continual import LwF

lwf = LwF(alpha=0.5, temperature=2.0)
old_logits = lwf.cache_old_outputs(inputs)
distillation_loss = lwf.compute_distillation_loss(new_logits, old_logits)
```

### 5. Hybrid Strategies

Combine multiple approaches:

```python
from cogniarch.continual import HybridStrategy

# EWC + Replay
hybrid = HybridStrategy(['ewc', 'replay'], {
    'ewc_lambda': 0.5,
    'replay_buffer_size': 500
})

# SI + Replay
hybrid = HybridStrategy(['si', 'replay'], {
    'si_xi': 0.1,
    'replay_buffer_size': 500
})
```

## Metrics

Evaluate continual learning performance:

```python
from cogniarch.continual import CLMetrics

metrics = CLMetrics()
metrics.record_accuracy(task_id=0, accuracy=0.95, at_training=True)
metrics.record_accuracy(task_id=0, accuracy=0.88, at_training=False)

print(f"Backward Transfer: {metrics.backward_transfer:.3f}")
print(f"Average Accuracy: {metrics.average_accuracy:.3f}")
print(f"Forgetting Measure: {metrics.forgetting_measure:.3f}")
```

### Key Metrics

- **Backward Transfer (BWT)**: Negative = forgetting, Positive = beneficial transfer
- **Forward Transfer (FWT)**: How much previous learning helps new tasks
- **Average Accuracy**: Mean accuracy across all tasks after final training
- **Forgetting Measure**: Maximum accuracy drop for any task

## Integration with Curriculum

```python
from cogniarch import CognitiveSystem, Curriculum
from cogniarch.continual import CLConfig

config = CLConfig(
    strategy='ewc_replay',
    lambda_=0.5,
    buffer_size=500,
    mode='task_incremental'
)

system = CognitiveSystem(cl_config=config)

curriculum = Curriculum()
curriculum.add_domain('math', 'Mathematics')
curriculum.add_lesson('math', 'counting_1_10', ...)
curriculum.add_lesson('math', 'counting_10_99', ...)

# Training automatically applies anti-forgetting
system.follow(curriculum)
system.teach('counting_1_10', data1)
system.teach('counting_10_99', data2)  # Won't forget counting 1-10
```

## Implementation Details

### Fisher Information Computation

```python
def compute_fisher_diagonal(model, data, loss_fn):
    """Compute diagonal of Fisher information matrix."""
    fisher = np.zeros_like(params)
    for x, y in data:
        loss = loss_fn(model.forward(params, x), y)
        grad = compute_gradient(loss, params)
        fisher += grad ** 2
    return fisher / len(data)
```

### Gradient Projection (GEM)

```python
def project_gradient(grad, stored_grads):
    """Project gradient to not increase previous task losses."""
    for stored_grad in stored_grads:
        if np.dot(grad, stored_grad) < 0:
            # Project onto half-space
            normal = stored_grad / np.linalg.norm(stored_grad)
            grad -= np.dot(grad, normal) * normal
    return grad
```

## Best Practices

1. **Choose Strategy Based on Task Similarity**:
   - Similar tasks → Replay-based methods
   - Dissimilar tasks → Architecture-based methods

2. **Tune Regularization Strength**:
   - Too low → Forgetting occurs
   - Too high → Plasticity lost (can't learn new tasks)

3. **Monitor Metrics**:
   - Track BWT to detect forgetting early
   - Adjust strategy if average accuracy drops

4. **Balance Buffer Size**:
   - Larger buffers improve retention but increase memory
   - Use reservoir sampling for unbounded streams

## API Reference

See individual strategy modules for complete API documentation.

## References

1. Kirkpatrick, J., et al. "Overcoming Catastrophic Forgetting in Neural Networks." PNAS (2017)
2. Zenke, F., et al. "Continual Learning Through Synaptic Intelligence." ICML (2017)
3. Lopez-Paz, D., Ranzato, M. "Gradient Episodic Memory." NeurIPS (2017)
4. Rusu, A.A., et al. "Progressive Neural Networks." arXiv (2016)
5. Li, Z., Hoiem, D. "Learning Without Forgetting." ECCV (2018)
