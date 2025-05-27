# llm_fairness_test.py
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
            rows.append({"prompt": prompt, "response": output, "group": group})

    df = pd.DataFrame(rows)

    dataset = Dataset(df, sensitive_features=["group"], cat_columns=["group"])

    class LLMWrapper:
        def predict(self, df):
            return df["response"].astype(str).tolist()

    wrapped_model = Model(model=LLMWrapper().predict, model_type="text_generation", name="LLM")

    print("Running Giskard scan for LLM fairness...")
    report = scan(wrapped_model, dataset)
    print("Giskard scan complete.")

    overall_llm_status = "Pass"
    if report.has_issues():
        overall_llm_status = "Fairness Violation (Giskard Issues Detected)"

    return {
        "test_type": "LLM fairness comparison (Giskard)",
        "overall_status": overall_llm_status,
        "giskard_report": report.to_dict()
    }
