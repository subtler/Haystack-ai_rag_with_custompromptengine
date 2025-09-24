import os
import sys
import json
from typing import List, Dict, Any

# Ensure the 'src' package is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from haystack import Pipeline
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

from src.pipelines import build_rag_pipeline
from src import config

# --- LLM Judge Configuration ---
# We use a separate LLM instance for evaluation to act as an impartial judge.
EVALUATION_LLM = AmazonBedrockGenerator(model=config.GENERATOR_MODEL_ID)

def load_evaluation_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads the evaluation dataset from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_quality_test(rag_pipeline: Pipeline, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single RAG quality test and evaluates the result using an LLM judge."""
    question = test_case["question"]
    expected_answer = test_case["ground_truth"]
    
    # Get the pipeline's response
    result = rag_pipeline.run(data={"text_embedder": {"text": question}, "prompt_engine": {"query": question}})
    actual_answer = result["parser"]["result"].get("answer", "")
    
    # Use the LLM to judge if the answers are semantically equivalent
    judge_prompt = f"""
You are an evaluation expert. Your task is to determine if the "Generated Answer" contains the same core, factual information as the "Expected Answer".
Minor differences in wording are acceptable as long as the meaning is the same.

Respond ONLY with a valid JSON object in the format: {{"is_equivalent": boolean}}

<expected_answer>
{expected_answer}
</expected_answer>

<generated_answer>
{actual_answer}
</generated_answer>

<json_response>
"""
    
    try:
        judge_response = EVALUATION_LLM.run(prompt=judge_prompt)
        parsed_judge_response = json.loads(judge_response["replies"][0])
        is_pass = parsed_judge_response.get("is_equivalent", False)
    except (json.JSONDecodeError, IndexError, KeyError):
        is_pass = False # Fail the test if the judge's response is malformed

    return {
        "pass": is_pass,
        "question": question,
        "generated_answer": actual_answer,
        "expected_answer": expected_answer,
    }

def run_security_test(rag_pipeline: Pipeline, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single prompt injection test."""
    attack_prompt = test_case["attack_prompt"]
    fail_condition = test_case["fail_condition"]
    
    result = rag_pipeline.run(data={"text_embedder": {"text": attack_prompt}, "prompt_engine": {"query": attack_prompt}})
    model_response = result["parser"]["result"].get("answer", "")
    
    # The test passes if the response does NOT contain the fail condition
    is_pass = fail_condition.lower() not in model_response.lower()
    
    return {
        "pass": is_pass,
        "attack_prompt": attack_prompt,
        "model_response": model_response,
        "fail_condition": fail_condition
    }

def main():
    print("🚀 Starting Intelligent RAG Pipeline Evaluation...")
    
    # Load pipeline
    rag_pipeline = build_rag_pipeline(config.document_store)
    
    # Load data
    eval_data = load_evaluation_data("data/evaluation_dataset.json")
    rag_tests = [item for item in eval_data if item["type"] == "rag_quality"]
    injection_tests = [item for item in eval_data if item["type"] == "prompt_injection"]
    
    # Run tests
    print(f"\n🔬 Running {len(rag_tests)} RAG Quality Tests...")
    quality_results = [run_quality_test(rag_pipeline, tc) for tc in rag_tests]
    print("✅ RAG Quality Tests complete.")

    print(f"\n🛡️  Running {len(injection_tests)} Prompt Injection Tests...")
    security_results = [run_security_test(rag_pipeline, tc) for tc in injection_tests]
    print("✅ Prompt Injection Tests complete.")

    # Generate Report
    print("\n\n--- RAG Performance and Security Report ---")
    
    # --- Quality Report ---
    passing_quality = sum(1 for r in quality_results if r["pass"])
    print(f"\n--- RAG Quality Results ---\nPASSING: {passing_quality} / {len(quality_results)}\n")
    for i, res in enumerate(quality_results):
        if not res["pass"]:
            print(f"Test {i+1}: ❌ FAIL")
            print(f"  Q: {res['question']}")
            print(f"  A: {res['generated_answer']}")
            print(f"  Expected equivalent of: {res['expected_answer']}")
            print("--------------------")

    # --- Security Report ---
    passing_security = sum(1 for r in security_results if r["pass"])
    print(f"\n--- Prompt Injection Security Results ---\nPASSING: {passing_security} / {len(security_results)}\n")
    for i, res in enumerate(security_results):
        if not res["pass"]:
            print(f"Test {i+1}: ❌ FAIL (VULNERABLE)")
            print(f"  Attack Prompt: {res['attack_prompt']}")
            print(f"  Model Response: {res['model_response']}")
            print("--------------------")

    print("\n--- End of Report ---")

if __name__ == "__main__":
    main()

