# llm_fairness_test.py (UPDATED with robust LLMWrapper.predict logic)
import yaml
import json
import requests
from giskard import Dataset, Model, scan
import pandas as pd
import os

def run_llm_fairness_test(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    prompts = config['prompts']
    api_key = os.getenv("OPENAI_API_KEY", config.get('api_key', ''))

    if not api_key or api_key == "your_openai_api_key_here":
        print("Warning: OpenAI API key not set. Set OPENAI_API_KEY or update llm_config.yaml.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    rows = []

    # --- Initial data collection (prompts and responses) ---
    # This part collects the initial sample responses from your LLM to create the starting dataset
    for group, prompt_list in prompts.items():
        for prompt in prompt_list:
            output = ""
            try:
                response = requests.post(
                    config['endpoint'],
                    headers=headers,
                    json={
                        "model": config['model'],
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=30
                )
                response.raise_for_status()
                output = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            except requests.exceptions.RequestException as e:
                print(f"API call failed for group '{group}' prompt '{prompt}': {e}")
                output = f"API_ERROR: {e}"
            except Exception as e:
                print(f"An unexpected error occurred for group '{group}', prompt '{prompt}': {e}")
                output = f"UNEXPECTED_ERROR: {e}"
            rows.append({"prompt": prompt, "response": output, "group": group})

    df = pd.DataFrame(rows)

    dataset = Dataset(
        df,
        target=None,  # No target column for LLM text generation
        column_types={
            "group": "category",
            "prompt": "text",
            "response": "text"
        }
    )

    # --- UPDATED LLMWrapper: Handles both existing responses and generating new ones ---
    class LLMWrapper:
        def predict(self, df: pd.DataFrame):
            # If the DataFrame coming from Giskard already has a 'response' column
            # (e.g., it's the original dataset), just return those responses.
            # We also check if all values are null, meaning it might be an empty column
            # in a generated dataset that needs new responses.
            if "response" in df.columns and not df["response"].isnull().all():
                return df["response"].astype(str).tolist()
            else:
                # If 'response' column is missing or empty, it means Giskard wants
                # us to generate new responses for the 'prompt' column.
                if "prompt" not in df.columns:
                    raise ValueError("DataFrame must contain a 'prompt' column for LLM input.")

                generated_responses = []
                for prompt_text in df["prompt"]:
                    llm_output = ""
                    try:
                        # Make API call to your LLM here to generate new responses
                        # Access config and headers from the outer scope (closure)
                        llm_response = requests.post(
                            config['endpoint'],
                            headers=headers,
                            json={
                                "model": config['model'],
                                "messages": [{"role": "user", "content": prompt_text}]
                            },
                            timeout=30
                        )
                        llm_response.raise_for_status()
                        llm_output = llm_response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    except requests.exceptions.RequestException as e:
                        print(f"LLMWrapper API call failed for prompt '{prompt_text}': {e}")
                        llm_output = f"API_ERROR: {e}"
                    except Exception as e:
                        print(f"LLMWrapper unexpected error for prompt '{prompt_text}': {e}")
                        llm_output = f"UNEXPECTED_ERROR: {e}"
                    generated_responses.append(llm_output)
                return generated_responses

    wrapped_model = Model(
        model=LLMWrapper().predict,
        model_type="text_generation",
        name="LLM Fairness Agent",
        description="Evaluates if an LLM generates biased responses based on demographic prompt differences.",
        feature_names=["prompt"],
        output_feature="response"
    )

    print("Running Giskard scan for LLM fairness...")
    # Keep raise_exceptions=True to get detailed error messages if detectors fail
    report = scan(wrapped_model, dataset, raise_exceptions=True)
    print("Giskard scan complete.")

    overall_llm_status = "Pass"
    if report.has_issues():
        overall_llm_status = "Fairness Violation (Giskard Issues Detected)"

    return {
        "test_type": "LLM fairness comparison (Giskard)",
        "overall_status": overall_llm_status,
        "giskard_report": json.loads(report.to_json())
    }