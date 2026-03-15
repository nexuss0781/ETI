
import numpy as np
from curriculum_engine import CurriculumEngine
import os
import logging


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-10)
    nb = float(np.linalg.norm(b) + 1e-10)
    return float(np.dot(a / na, b / nb))


def _encode_number(engine: CurriculumEngine, n: int, max_digits: int) -> np.ndarray:
    tokens = list(str(int(n)))
    return engine.encoder.encode_sequence(tokens, max_len=max_digits)


def _encode_object(engine: CurriculumEngine, max_digits: int) -> np.ndarray:
    return engine.encoder.encode_sequence(["OBJECT"], max_len=max_digits)


def _predict_vec(engine: CurriculumEngine, x: np.ndarray) -> np.ndarray:
    if engine._cl_system is None:
        raise RuntimeError("No trained CL system found. Train at least one lesson before querying.")
    in_dim = int(engine._cl_system.config.input_dim)
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.shape[0] < in_dim:
        x = np.pad(x, (0, in_dim - x.shape[0]))
    else:
        x = x[:in_dim]
    params = engine._cl_system.learner.params
    y = engine._cl_system.base_lens.forward(params.data, x)
    return np.asarray(y, dtype=np.float64).flatten()


def _decode_number_nearest(
    engine: CurriculumEngine,
    vec: np.ndarray,
    candidates: list[int],
    max_digits: int,
) -> tuple[int, float]:
    best_n = candidates[0]
    best_s = -1.0
    v = np.asarray(vec, dtype=np.float64).flatten()
    for n in candidates:
        c = _encode_number(engine, n, max_digits)
        s = _cosine(v, c)
        if s > best_s:
            best_s = s
            best_n = n
    return best_n, best_s


def _eval_successor(
    engine: CurriculumEngine,
    test_numbers: list[int],
    decode_candidates: list[int],
    max_digits: int,
    title: str,
) -> dict:
    correct = 0
    sims: list[float] = []
    errors: list[tuple[int, int, int, float]] = []
    for n in test_numbers:
        x = _encode_number(engine, n, max_digits)
        y_pred = _predict_vec(engine, x)
        pred_n, sim = _decode_number_nearest(engine, y_pred, decode_candidates, max_digits)
        target = int(n + 1)
        ok = pred_n == target
        correct += 1 if ok else 0
        sims.append(float(sim))
        if not ok and len(errors) < 15:
            errors.append((int(n), target, int(pred_n), float(sim)))
    acc = correct / max(1, len(test_numbers))
    avg_sim = float(np.mean(sims)) if sims else 0.0
    print(f"\n[{title}]")
    print(f"  N={len(test_numbers)}  accuracy={acc:.4f}  avg_decode_sim={avg_sim:.4f}")
    if errors:
        print("  sample_errors (n, target, pred, sim):")
        for e in errors:
            print(f"    {e[0]:>6d} -> {e[1]:>6d} | pred={e[2]:>6d} sim={e[3]:.3f}")
    return {"title": title, "n": len(test_numbers), "accuracy": acc, "avg_decode_sim": avg_sim}

# Setup logging to be less verbose
logging.getLogger("eti.continual_learning").setLevel(logging.WARNING)
logging.getLogger("eti.curriculum").setLevel(logging.INFO)

def train():
    print("Initializing Curriculum Engine for STEP 2: MATHS...")
    # 1. Create engine (loads previous state from eti_memory)
    engine = CurriculumEngine.create(
        memory_dir="eti_memory",
        cl_strategy="ewc_replay", # Using replay for better retention of Step 1
        hidden_dims=[256, 256],    # Increased capacity for number counting
        learning_rate=0.003,
        epochs_per_lesson=100,
        verbose=True,
    )

    max_digits = 6

    # 2. Define the Math domain
    if "math" not in engine.domains:
        engine.new_domain("math", "Mathematics", "Number counting and sequential progression")

    # 3. Define Rules
    # Rule 1: Numbers are objects (Bridge from Step 1)
    engine.define_rule(
        rule_id="number_is_object",
        name="Number is Object",
        domain="math",
        pattern="[number] = OBJECT",
        weight=1.5,
    )
    
    # Rule 2: Successor rule (Strict sequential progress)
    engine.define_rule(
        rule_id="successor_rule",
        name="Successor Rule",
        domain="math",
        pattern="[n] -> [n+1]",
        weight=2.0, # Foundational rule for this step
    )

    # 4. Create Lessons
    # Lesson A: Numbers are Objects
    engine.new_lesson(
        lesson_id="math_num_obj",
        name="Numbers as Objects",
        domain="math",
        rule_ids=["number_is_object"],
        epochs=50,
        competency_threshold=0.90
    )

    # Lesson B: Basic Step-by-Step Counting (1-10)
    engine.new_lesson(
        lesson_id="math_count_1_10",
        name="Counting 1 to 10",
        domain="math",
        rule_ids=["successor_rule"],
        epochs=100,
        competency_threshold=0.85,
        prerequisite_ids=["math_num_obj"]
    )

    # Lesson C: Scaling 10-99
    engine.new_lesson(
        lesson_id="math_count_10_99",
        name="Counting 10 to 99",
        domain="math",
        rule_ids=["successor_rule"],
        epochs=20,
        competency_threshold=0.80,
        prerequisite_ids=["math_count_1_10"]
    )

    # 5. Data Generators
    def num_obj_gen():
        data = []
        target_vec = _encode_object(engine, max_digits)
        # Sample numbers up to 100 to establish the class
        for i in range(1, 101):
            input_vec = _encode_number(engine, i, max_digits)
            data.append((input_vec, target_vec))
        return data

    def count_1_10_gen():
        data = []
        # Strict sequence: 1->2, 2->3, ..., 9->10
        for i in range(1, 10):
            input_vec = _encode_number(engine, i, max_digits)
            target_vec = _encode_number(engine, i + 1, max_digits)
            data.append((input_vec, target_vec))
        # Add some noise/repetitions to help generalization
        return data * 20

    def count_10_99_gen():
        data = []
        # Sequential progression: 10->11, ..., 98->99
        for i in range(10, 100):
            input_vec = _encode_number(engine, i, max_digits)
            target_vec = _encode_number(engine, i + 1, max_digits)
            data.append((input_vec, target_vec))
        return data * 5

    def count_100_999_gen():
        data = []
        for i in range(100, 1000):
            input_vec = _encode_number(engine, i, max_digits)
            target_vec = _encode_number(engine, i + 1, max_digits)
            data.append((input_vec, target_vec))
        return data * 2

    if engine._find_lesson("math_count_100_999") is None:
        engine.new_lesson(
            lesson_id="math_count_100_999",
            name="Counting 100 to 999",
            domain="math",
            rule_ids=["successor_rule"],
            epochs=20,
            competency_threshold=0.80,
            prerequisite_ids=["math_count_10_99"],
        )

    # 6. Generate data and Teach
    print("\n--- Lesson 1: Numbers are Objects ---")
    engine.generate_lesson_data("math_num_obj", custom_generator=num_obj_gen)
    engine.teach_lesson("math_num_obj")

    print("\n--- Lesson 2: Step-by-Step Counting (1-10) ---")
    engine.generate_lesson_data("math_count_1_10", custom_generator=count_1_10_gen)
    engine.teach_lesson("math_count_1_10")

    print("\n--- Lesson 3: Scaling 10-99 ---")
    engine.generate_lesson_data("math_count_10_99", custom_generator=count_10_99_gen)
    engine.teach_lesson("math_count_10_99")

    print("\n--- Lesson 4: Scaling 100-999 ---")
    engine.generate_lesson_data("math_count_100_999", custom_generator=count_100_999_gen)
    engine.teach_lesson("math_count_100_999")

    # 7. Evaluation (successor / strict progression)
    decode_candidates = list(range(1, 2000)) + [10**k for k in range(1, 7)]
    decode_candidates = sorted(set(decode_candidates))

    _eval_successor(
        engine,
        test_numbers=list(range(1, 101)),
        decode_candidates=decode_candidates,
        max_digits=max_digits,
        title="Successor test: 1..100",
    )

    _eval_successor(
        engine,
        test_numbers=[10, 20, 50, 99, 100, 101, 110, 150, 199, 200, 250, 499, 500, 999],
        decode_candidates=decode_candidates,
        max_digits=max_digits,
        title="Successor test: mixed seen/unseen up to 999",
    )

    _eval_successor(
        engine,
        test_numbers=[10**k for k in range(1, 6)],
        decode_candidates=decode_candidates,
        max_digits=max_digits,
        title="Successor test: powers of 10",
    )

    _eval_successor(
        engine,
        test_numbers=[1234, 1999],
        decode_candidates=decode_candidates,
        max_digits=max_digits,
        title="Extrapolation test (within max_digits): 1234, 1999",
    )

    # 8. Final Save
    engine.save()
    print("\nSTEP 2 Training complete and state saved to eti_memory.")

if __name__ == "__main__":
    train()
