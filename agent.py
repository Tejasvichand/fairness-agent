# agent.py
import argparse
import json
from fairness_test import run_fairness_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fairness Test Agent")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    report = run_fairness_test(args.config)
    with open("results/fairness_results.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Fairness test complete. Results saved to results/fairness_results.json")
