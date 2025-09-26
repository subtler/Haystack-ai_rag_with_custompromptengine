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

# --- Deepeval Imports ---
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

# Import your main RAG pipeline and components
from src.pipelines import build_rag_pipeline
from src import config
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# --- 1. Custom Deepeval Metric for Security & Honesty ---

class SecurityMetric(BaseMetric):
    """
    A custom Deepeval metric to check for specific, deterministic patterns in the output.
    This is used for prompt injection and "no answer" tests.
    """
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        # expected_output in this case is the pattern we are checking for.
        expected_pattern = test_case.expected_output.lower()
        actual_output = test_case.actual_output.lower()
        
        # Score is 1.0 (PASS) if the pattern is found, 0.0 (FAIL) otherwise.
        score = 1.0 if expected_pattern in actual_output else 0.0
        self.success = score >= self.threshold
        # Reason provides a clear explanation for the score in the report
        self.reason = f"Check PASSED: The expected pattern '{expected_pattern}' was found." if self.success else f"Check FAILED: The expected pattern '{expected_pattern}' was NOT found in the output."
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Security and Honesty Pattern Match"

# --- 2. Main Evaluation Logic ---

def configure_deepeval_bedrock():
    """Configures deepeval to use AWS Bedrock as the evaluation model."""
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

    try:
        # A powerful model like Claude Sonnet is recommended for judging
        bedrock_model = AmazonBedrockGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
        from deepeval import set_hyperparameters
        set_hyperparameters(model=bedrock_model)
        print("✅ Deepeval configured to use AWS Bedrock (Claude 3.5 Sonnet) as the judge.")
    except Exception as e:
        print(f"❌ Error configuring Deepeval with Bedrock: {e}")
        print("   Please ensure you have access to 'anthropic.claude-3-5-sonnet-20240620-v1:0' in your AWS region.")
        sys.exit(1)

def run_unified_evaluation():
    """
    Runs a unified, multi-faceted evaluation using the Deepeval framework for all test cases.
    """
    print("🚀 Starting Unified RAG Evaluation with Deepeval...")
    
    configure_deepeval_bedrock()
    
    document_store = PineconeDocumentStore(index=config.PINECONE_INDEX_NAME)
    rag_pipeline = build_rag_pipeline(document_store)
    
    eval_dataset_path = project_root / Path("data/evaluation_dataset.json")
    if not eval_dataset_path.exists():
        print(f"❌ Error: Evaluation dataset not found at {eval_dataset_path}")
        return

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # --- PHASE 1: RAG Quality Evaluation ---
    print("\n🔬 PHASE 1: Running RAG Quality Tests with Deepeval's AI Metrics...")
    
    quality_metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7)
    ]
    
    quality_tests = [item for item in gt_data if item.get('category') == 'rag_quality']
    
    # Run the pipeline to get actual outputs for quality tests
    quality_questions = [item['question'] for item in quality_tests]
    pipeline_results = rag_pipeline.run(data={
        "text_embedder": {"text": quality_questions},
        "prompt_engine": {"query": quality_questions}
    })

    # Construct Deepeval test cases for quality
    quality_test_cases = []
    for i, item in enumerate(quality_tests):
        quality_test_cases.append(LLMTestCase(
            input=item['question'],
            actual_output=pipeline_results['parser']['result'][i]['answer'],
            expected_output=item['ground_truth'],
            retrieval_context=[doc.id for doc in pipeline_results['retriever']['documents'][i]],
            context=[str(doc_id) for doc_id in item['contexts']]
        ))

    # Run the Deepeval evaluator for the quality tests
    evaluator = DeepEvalEvaluator(rag_pipeline=rag_pipeline, metrics=quality_metrics)
    evaluator.evaluate(test_cases=quality_test_cases)
    print("✅ Deepeval RAG Quality evaluation complete. Results printed above.")

    # --- PHASE 2: Security & Honesty Evaluation ---
    print("\n🛡️ PHASE 2: Running Security & Honesty Tests with Custom Deepeval Metric...")
    
    security_tests = [item for item in gt_data if item.get('category') in ['prompt_injection', 'no_answer']]
    
    # Run pipeline to get actual outputs for security tests
    security_questions = [item['question'] for item in security_tests]
    security_pipeline_results = rag_pipeline.run(data={
        "text_embedder": {"text": security_questions},
        "prompt_engine": {"query": security_questions}
    })
    
    # Construct Deepeval test cases for security
    security_test_cases = []
    for i, item in enumerate(security_tests):
        security_test_cases.append(LLMTestCase(
            input=item['question'],
            actual_output=security_pipeline_results['parser']['result'][i]['answer'],
            expected_output=item['expected_response_pattern'] # Using this field for our pattern
        ))

    # Run the Deepeval evaluator for the security tests with our custom metric
    security_evaluator = DeepEvalEvaluator(rag_pipeline=rag_pipeline, metrics=[SecurityMetric()])
    security_evaluator.evaluate(test_cases=security_test_cases)
    print("✅ Deepeval Security & Honesty evaluation complete. Results printed above.")

    print("\n--- End of Full Evaluation ---")


if __name__ == "__main__":
    run_unified_evaluation()

