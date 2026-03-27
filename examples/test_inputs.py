import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.langgraph_pipeline import pipeline_app

def run_test(test_name, text):
    print(f"\n{'='*60}\n{test_name}\n{'='*60}")
    print(f"Input Text: '{text}'\n")
    
    result = pipeline_app.invoke({"text": text})
    
    print(f"\n--- FINAL RESULT ---")
    print(f"Prediction: {result.get('prediction')}")
    print(f"Confidence: {result.get('confidence', 0.0)*100:.1f}%")
    print(f"Source:     {result.get('source')}")
    if result.get('error'):
        print(f"Notes:      {result.get('error')}")

if __name__ == "__main__":
    # Test 1: High Confidence (> 0.80) -> ACCEPT
    run_test("TEST 1: HIGH CONFIDENCE (>80%)", "I absolutely love this product, it is amazing!")

    # Test 2: Medium Confidence (0.50 - 0.80) -> CLARIFY
    run_test("TEST 2: MEDIUM CONFIDENCE (50%-80%)", "DEMO_CLARIFY")

    # Test 3: Low Confidence (< 0.50) -> FALLBACK
    run_test("TEST 3: LOW CONFIDENCE (<50%)", "DEMO_FALLBACK")
    
    # Test 4: System Crash / Error -> FALLBACK
    run_test("TEST 4: SYSTEM CRASH (Text too long)", "This is way too long to process. " * 500)