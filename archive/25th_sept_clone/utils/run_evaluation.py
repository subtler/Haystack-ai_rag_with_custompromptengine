# utils/run_evaluation.py

import sys
import os
import json
from pathlib import Path
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# --- Add Project Root to the Path ---
# This allows the script to find the 'src' package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from our source package
from src import config
from src.pipelines import build_rag_pipeline

def run_evaluation():
    """
    Runs a comprehensive evaluation of the RAG pipeline using a structured dataset.
    This version is updated to work with a pipeline that produces Pydantic-validated JSON.
    """
    print("🚀 Starting Pydantic-Compatible RAG Pipeline Evaluation...")

    # --- 1. Load the RAG Pipeline ---
    document_store = PineconeDocumentStore(index=config.PINECONE_INDEX_NAME)
    rag_pipeline = build_rag_pipeline(document_store)
    
    # --- 2. Load the Evaluation Dataset ---
    dataset_path = Path(project_root) / "data" / "evaluation_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    # --- KEY CHANGE: Use .get() for safe dictionary access to prevent KeyError ---
    quality_tests = [item for item in evaluation_data if item.get('category') == 'rag_quality']
    injection_tests = [item for item in evaluation_data if item.get('category') == 'prompt_injection']
    
    quality_pass_count = 0
    injection_pass_count = 0

    # --- 3. Run RAG Quality Tests ---
    print(f"\n🔬 Running {len(quality_tests)} RAG Quality Tests...")
    quality_results_log = []
    for i, test in enumerate(quality_tests):
        question = test["question"]
        expected_substring = test["ground_truth"]
        
        # Run the pipeline
        result = rag_pipeline.run(
            data={
                "text_embedder": {"text": question},
                "prompt_engine": {"query": question}
            }
        )
        
        # Access the validated JSON output
        pipeline_output = result["parser"]["result"]
        actual_answer = pipeline_output.get("answer", "")

        # A simple check for correctness
        passed = expected_substring.lower() in actual_answer.lower()
        if passed:
            quality_pass_count += 1
        
        result_str = "✅ PASS" if passed else "❌ FAIL"
        log_entry = (
            f"Test {i+1}: {result_str}\n"
            f"  Q: {question}\n"
            f"  A: {actual_answer}\n"
            f"  Expected to contain: {expected_substring}"
        )
        quality_results_log.append(log_entry)
    print("✅ RAG Quality Tests complete.")

    # --- 4. Run Prompt Injection Tests ---
    print(f"\n🛡️  Running {len(injection_tests)} Prompt Injection Tests...")
    injection_results_log = []
    for i, test in enumerate(injection_tests):
        attack_prompt = test["question"]
        failure_condition = test["ground_truth"] # In these tests, ground_truth is the failure string
        
        result = rag_pipeline.run(
            data={
                "text_embedder": {"text": attack_prompt},
                "prompt_engine": {"query": attack_prompt}
            }
        )
        
        pipeline_output = result["parser"]["result"]
        model_response = pipeline_output.get("answer", "")
        
        # Test passes if the failure condition is NOT met
        vulnerable = failure_condition.lower() in model_response.lower()
        passed = not vulnerable
        
        if passed:
            injection_pass_count += 1

        result_str = "✅ PASS" if passed else "❌ FAIL (VULNERABLE)"
        log_entry = (
            f"Test {i+1}: {result_str}\n"
            f"  Attack Prompt: {attack_prompt}\n"
            f"  Model Response: {model_response}"
        )
        injection_results_log.append(log_entry)
    print("✅ Prompt Injection Tests complete.")

    # --- 5. Print Final Report ---
    print("\n\n--- RAG Performance and Security Report ---")
    
    print("\n--- RAG Quality Results ---")
    print(f"PASSING: {quality_pass_count} / {len(quality_tests)}\n")
    for log in quality_results_log:
        print(log)
        print("--------------------")

    print("\n--- Prompt Injection Security Results ---")
    print(f"PASSING: {injection_pass_count} / {len(injection_tests)}\n")
    for log in injection_results_log:
        print(log)
        print("--------------------")

    print("\n--- End of Report ---")

if __name__ == "__main__":
    run_evaluation()

