# agent.py
import argparse
import json
from fairness_test import run_fairness_test
from llm_fairness_test import run_llm_fairness_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fairness Test Agent")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--llm", action="store_true", help="Run LLM fairness test")
    args = parser.parse_args()

    if args.llm:
        report = run_llm_fairness_test(args.config)
        result_file = "results/llm_fairness_results.json"
    else:
        report = run_fairness_test(args.config)
        result_file = "results/fairness_results.json"

    with open(result_file, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Fairness test complete. Results saved to {result_file}")
