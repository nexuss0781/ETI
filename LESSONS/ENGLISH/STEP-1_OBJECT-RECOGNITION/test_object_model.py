
import numpy as np
from curriculum_engine import CurriculumEngine
import re

def test():
    # 1. Load the engine
    engine = CurriculumEngine.create(memory_dir="eti_memory", verbose=False)
    
    # 2. Real-world test paragraph (Amharic)
    paragraph = "በጦር ካምፕ ውስጥ አበበ የሚያደርጋቸውን ልምምዶች እና ውድድሮች በጥሞና ይከታተሉ የነበሩት ስዊድናዊ አሰልጣኝ ኦኒ ኒስካነን የአበበን ችሎታ በመገንዘባቸው ከሌሎቹ አትሌቶች ጋር ተቀላቅሎ ልምምድ እንዲሠራ ያደርጉት እና የሮም ኦሎምፒክ እስኪቃረብ ድረስ በሚል ከሁለት ሺህ ሜትር በላይ ከፍታ ባላቸው ተራራማ ቦታዎች ላይ የመሮጥ እና የሺህ አምስት መቶ ሜትር ተደጋጋሚ የፍጥነት ሩጫ ልምምዶችን እንዲሠሩ ያደርጓቸዋል።"
    
    # Identify all objects (tokens) in the paragraph
    tokens = re.findall(r'\w+|[^\w\s]', paragraph, re.UNICODE)
    
    print("--- REAL WORLD TEST: OBJECT IDENTIFICATION ---")
    print(f"Paragraph length: {len(paragraph)} characters")
    print(f"Total tokens identified: {len(tokens)}")
    
    # Target representation of "OBJECT"
    target_vec = engine.encoder.encode("OBJECT")
    
    # Query the model for each token
    object_count = 0
    for token in tokens:
        x = engine.encoder.encode(token)
        # The model's forward pass should return something close to target_vec
        params = engine._cl_system.learner.params
        pred = engine._cl_system.base_lens.forward(params.data, x)
        
        # Calculate cosine similarity to the "OBJECT" representation
        sim = np.dot(pred, target_vec) / (np.linalg.norm(pred) * np.linalg.norm(target_vec) + 1e-10)
        
        if sim > 0.9: # High similarity means it's an object
            object_count += 1
            
    print(f"Model identified {object_count}/{len(tokens)} tokens as OBJECTS (via neural representation).")
    
    # 3. Character distinctness test: ሩ vs ጫ
    print("\n--- CHARACTER DISTINCTNESS TEST ---")
    char1 = "ሩ"
    char2 = "ጫ"
    
    vec1 = engine.encoder.encode(char1)
    vec2 = engine.encoder.encode(char2)
    
    # Check if they are distinct objects
    dist = np.linalg.norm(vec1 - vec2)
    print(f"Object '{char1}' vector (first 4): {vec1[:4]}")
    print(f"Object '{char2}' vector (first 4): {vec2[:4]}")
    print(f"Euclidean distance between '{char1}' and '{char2}': {dist:.4f}")
    
    if dist > 1e-5:
        print(f"SUCCESS: '{char1}' IS NOT SIMILAR TO '{char2}'")
    else:
        print(f"FAILURE: '{char1}' and '{char2}' are too similar")
        
    # Check identity: ሩ vs ሩ
    vec1_again = engine.encoder.encode(char1)
    dist_identity = np.linalg.norm(vec1 - vec1_again)
    if dist_identity < 1e-10:
        print(f"SUCCESS: '{char1}' IS SAME AS '{char1}'")
    else:
        print(f"FAILURE: Identity check failed for '{char1}'")

    # 4. Final fine-tuning requirement: return "IT IS OBJECT BY ITSELF"
    print("\n--- FINAL REPRESENTATION TEST ---")
    test_token = "አበበ"
    x_test = engine.encoder.encode(test_token)
    params = engine._cl_system.learner.params
    pred_test = engine._cl_system.base_lens.forward(params.data, x_test)
    sim_test = np.dot(pred_test, target_vec) / (np.linalg.norm(pred_test) * np.linalg.norm(target_vec) + 1e-10)
    
    if sim_test > 0.9:
        print(f"Input '{test_token}' -> IT IS OBJECT BY ITSELF")

if __name__ == "__main__":
    test()
