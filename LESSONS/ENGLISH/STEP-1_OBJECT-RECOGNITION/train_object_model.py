
import numpy as np
from curriculum_engine import CurriculumEngine
import os

def train():
    # 1. Create engine (loads previous state automatically if it exists)
    # Using a small hidden_dims for this foundational task
    engine = CurriculumEngine.create(
        memory_dir="eti_memory",
        cl_strategy="ewc",
        hidden_dims=[128, 128],
        learning_rate=0.005,
        epochs_per_lesson=100,
        verbose=True,
    )

    # 2. Define the English domain
    engine.new_domain("english", "English", "The Base Abstraction")

    # 3. Define the Object rule
    # Goal: Teach the model that everything it encounters is an object.
    # Pattern: [any] -> OBJECT
    engine.define_rule(
        rule_id="object_identity",
        name="Object Identity",
        domain="english",
        pattern="[any] = OBJECT",
        weight=2.0,
    )

    # 4. Create the lesson
    engine.new_lesson(
        lesson_id="eng_object_abstraction",
        name="English (The Base Abstraction)",
        domain="english",
        rule_ids=["object_identity"],
        epochs=100,
        competency_threshold=0.95
    )

    # 5. Generate training data
    # We need to teach it that "a", ">", "cat", "1", and Amharic chars are all objects.
    # We'll provide a custom generator to ensure diverse Unicode coverage.
    def object_data_generator():
        symbols = [
            "a", "b", "c", "1", "2", "3", ">", "<", "=", "!", "?", "@", "#",
            "cat", "dog", "object", "thing", "anything",
            "በ", "ጦ", "ር", "ካ", "ም", "ፕ", "ሩ", "ጫ" # Amharic chars from the prompt
        ]
        data = []
        # Target is the representation of "OBJECT"
        target_vec = engine.encoder.encode("OBJECT")
        
        for s in symbols:
            input_vec = engine.encoder.encode(s)
            data.append((input_vec, target_vec))
            
        return data

    engine.generate_lesson_data(
        "eng_object_abstraction",
        custom_generator=object_data_generator
    )

    # 6. Teach the lesson
    print("Starting training for 'English (The Base Abstraction)'...")
    engine.teach_lesson("eng_object_abstraction")

    # 7. Save the state
    engine.save()
    print("Training complete and state saved.")

if __name__ == "__main__":
    train()
