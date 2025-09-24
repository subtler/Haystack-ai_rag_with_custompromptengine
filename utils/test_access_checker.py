import os
import boto3
import json
from dotenv import load_dotenv

# --- List of Models to Test ---
# These are the models we recommended in the production strategy.
# The script will try to invoke each one.
MODELS_TO_TEST = [
    # Tier 1: Core Production Models
    "amazon.titan-embed-text-v2:0",
    "anthropic.claude-3-haiku-20240307-v1:0",

    # Tier 2: High-Performance Models
    "cohere.embed-english-v3",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",

    # Tier 3: Strategic Alternative
    "us.meta.llama3-2-11b-instruct-v1:0",
]

def get_boto3_client():
    """Initializes and returns a boto3 client for Bedrock."""
    load_dotenv()
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    # Boto3 will automatically find system-configured credentials
    bedrock_client = boto3.client(
        service_name="bedrock-runtime", # Use 'bedrock-runtime' for invoking models
        region_name=aws_region,
    )
    return bedrock_client

def test_model_access(bedrock_client, model_id):
    """
    Tries to invoke a specific Bedrock model to check for access.
    Returns True if successful, False otherwise.
    """
    print(f"-> Testing Model ID: {model_id}...")
    
    try:
        # Construct a minimal, valid payload for the model
        if "amazon.titan-embed" in model_id:
            body = json.dumps({"inputText": "test"})
        elif "cohere.embed" in model_id:
            body = json.dumps({"texts": ["test"], "input_type": "search_document"})
        elif "anthropic.claude" in model_id:
            # Claude 3 requires the new messages format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hello"}]
            })
        elif "meta.llama" in model_id:
            body = json.dumps({
                "prompt": "hello",
                "max_gen_len": 10
            })
        else:
            print("   ⚠️  WARN: No test payload configured for this model type. Skipping invocation.")
            return False, "Skipped"

        # Invoke the model
        response = bedrock_client.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        
        # A successful call (even if the content is empty) means we have access
        return True, "Success"

    except Exception as e:
        # Specifically check for the access denied error
        if "AccessDeniedException" in str(e):
            return False, "Access Denied"
        else:
            return False, f"An unexpected error occurred: {e}"

def main():
    """Main function to test access for a list of Bedrock models."""
    print("🚀 Starting AWS Bedrock Model Access Test...")
    
    try:
        client = get_boto3_client()
        print(f"✅ Successfully connected to Bedrock Runtime in region '{client.meta.region_name}'.\n")
    except Exception as e:
        print(f"❌ Failed to create Bedrock client. Error: {e}")
        print("   Please ensure your AWS credentials and region are configured correctly.")
        return

    results = {}
    for model_id in MODELS_TO_TEST:
        is_accessible, reason = test_model_access(client, model_id)
        results[model_id] = {"accessible": is_accessible, "reason": reason}

    print("\n--- Test Results Summary ---")
    all_successful = True
    for model_id, result in results.items():
        if result["accessible"]:
            print(f"✅ SUCCESS: You have access to '{model_id}'.")
        else:
            all_successful = False
            print(f"❌ FAILURE: You do NOT have access to '{model_id}'. Reason: {result['reason']}")
            
    if all_successful:
        print("\n🎉 Congratulations! You have access to all the requested models.")
    else:
        print("\nAction Required: Please request access for the failed models in the AWS Bedrock console.")


if __name__ == "__main__":
    main()