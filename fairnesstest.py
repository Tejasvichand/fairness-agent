# fairness_test.py
import pandas as pd
import joblib
import yaml
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

def run_fairness_test(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['data_path'])
    model = joblib.load(config['model_path'])

    X = df[config['features']]
    y_pred = model.predict(X)
    df['prediction'] = y_pred

    group = df[config['protected_attribute']]
    metric = MetricFrame(metrics=selection_rate, y_pred=df['prediction'], sensitive_features=group)
    disparity = demographic_parity_difference(df['prediction'], sensitive_features=group)

    return {
        "protected_attribute": config['protected_attribute'],
        "groups": list(group.unique()),
        "selection_rate_per_group": metric.by_group.to_dict(),
        "statistical_parity_gap": disparity,
        "threshold": config['threshold'],
        "status": "Fairness Violation" if abs(disparity) > config['threshold'] else "Pass"
    }
