# rag_evals/run_deepeval_evaluation.py

import sys
import os
import json
from pathlib import Path
from typing import List

# --- Add the project's root directory to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Haystack Imports ---
from haystack.dataclasses import Document

# --- Deepeval Imports (using the correct, modern integration path) ---
from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator
from deepeval.metrics import BaseMetric, FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# Import your main RAG pipeline and components
from src.pipelines import build_rag_pipeline
from src import config
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# --- 1. Custom Deepeval Metric for Security & Honesty ---
# This remains the same, as it's a powerful way to define custom checks.
class SecurityMetric(BaseMetric):
    """A custom Deepeval metric to check for specific, deterministic patterns in the output."""
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        expected_pattern = test_case.expected_output.lower()
        actual_output = test_case.actual_output.lower()
        score = 1.0 if expected_pattern in actual_output else 0.0
        self.success = score >= self.threshold
        self.reason = f"Check PASSED: The expected pattern '{expected_pattern}' was found." if self.success else f"Check FAILED: The expected pattern '{expected_pattern}' was NOT found."
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Security and Honesty Pattern Match"

# --- 2. Main Evaluation Logic ---

def configure_deepeval_bedrock():
    """Configures deepeval to use the AWS Bedrock model specified in the config file."""
    from deepeval.models.bedrock_model import BedrockModel
    try:
        bedrock_model = BedrockModel(model=config.EVALUATION_MODEL_ID)
        from deepeval import set_hyperparameters
        set_hyperparameters(model=bedrock_model)
        print(f"✅ Deepeval configured to use AWS Bedrock ({config.EVALUATION_MODEL_ID}) as the judge.")
    except Exception as e:
        print(f"❌ Error configuring Deepeval with Bedrock: {e}")
        sys.exit(1)

def run_unified_evaluation():
    """Runs an efficient, unified evaluation using the official DeepEval Haystack integration."""
    print("🚀 Starting Unified RAG Evaluation with Deepeval...")
    
    configure_deepeval_bedrock()
    
    document_store = PineconeDocumentStore(index=config.PINECONE_INDEX_NAME)
    rag_pipeline = build_rag_pipeline(document_store)
    
    eval_dataset_path = project_root / Path("data/evaluation_dataset.json")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # --- EFFICIENT PIPELINE EXECUTION: Run the RAG pipeline ONCE for all tests ---
    print("\n⚙️ Running RAG pipeline on all test cases...")
    all_questions = [item['question'] for item in gt_data]
    pipeline_results = rag_pipeline.run(data={
        "text_embedder": {"text": all_questions},
        "prompt_engine": {"query": all_questions}
    })
    print("✅ Pipeline execution complete.")

    # --- Sort the results for the evaluators ---
    quality_inputs = {"questions": [], "predicted_answers": [], "ground_truth_answers": [], "retrieved_documents": [], "contexts": []}
    security_test_cases = []

    for i, item in enumerate(gt_data):
        category = item.get('category')
        actual_answer_dict = pipeline_results['parser']['result'][i]
        
        if category == 'rag_quality':
            quality_inputs["questions"].append(item['question'])
            quality_inputs["predicted_answers"].append(actual_answer_dict)
            quality_inputs["ground_truth_answers"].append(item['ground_truth'])
            quality_inputs["retrieved_documents"].append(pipeline_results['retriever']['documents'][i])
            quality_inputs["contexts"].append([Document(id=str(doc_id)) for doc_id in item['contexts']])
        elif category in ['prompt_injection', 'no_answer', 'specificity']:
            security_test_cases.append(LLMTestCase(
                input=item['question'],
                actual_output=actual_answer_dict['answer'],
                expected_output=item.get('expected_response_pattern') or item.get('ground_truth')
            ))

    # --- PHASE 1: RAG Quality Evaluation ---
    print("\n🔬 PHASE 1: Analyzing RAG Quality Results...")
    if quality_inputs["questions"]:
        quality_metrics = [
            FaithfulnessMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7)
        ]
        # Use the official DeepEvalEvaluator component
        evaluator = DeepEvalEvaluator(metrics=quality_metrics)
        eval_results = evaluator.run(**quality_inputs)
        print(eval_results["results"]) # DeepevalEvaluator returns a results dictionary
        print("✅ Deepeval RAG Quality evaluation complete.")
    else:
        print("- No RAG Quality test cases found to evaluate.")

    # --- PHASE 2: Security & Honesty Evaluation ---
    print("\n🛡️ PHASE 2: Analyzing Security & Honesty Results...")
    if security_test_cases:
        # For our custom metric, we still need to wrap it in the evaluator
        security_evaluator = DeepEvalEvaluator(metrics=[SecurityMetric()])
        
        # We need to format the inputs for the security evaluator run
        security_inputs = {
            "test_cases": security_test_cases,
            # Provide dummy lists for unused parameters to satisfy the component
            "questions": [tc.input for tc in security_test_cases],
            "predicted_answers": [{"answer": tc.actual_output, "references": []} for tc in security_test_cases]
        }
        
        security_results = security_evaluator.run(**security_inputs)
        print(security_results["results"])
        print("✅ Deepeval Security & Honesty evaluation complete.")
    else:
        print("- No Security or Honesty test cases found to evaluate.")

    print("\n--- End of Full Evaluation ---")

if __name__ == "__main__":
    run_unified_evaluation()

