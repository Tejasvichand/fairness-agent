import pandas as pd
import numpy as np # For handling potential NaN values gracefully

class FairnessChecker:
    def __init__(self, df: pd.DataFrame, prediction_col: str, protected_attr: str, positive_label=1, reference_group=None):
        self.df = df.copy()
        self.prediction_col = prediction_col
        self.protected_attr = protected_attr
        self.positive_label = positive_label
        self.groups = df[protected_attr].unique()
        self.reference_group = reference_group

        # Basic validation for protected attribute and prediction column
        if self.protected_attr not in self.df.columns:
            raise ValueError(f"Protected attribute '{self.protected_attr}' not found in DataFrame.")
        if self.prediction_col not in self.df.columns:
            raise ValueError(f"Prediction column '{self.prediction_col}' not found in DataFrame.")

        # Ensure at least two groups for meaningful comparison
        if len(self.groups) < 2:
            raise ValueError(f"Protected attribute '{self.protected_attr}' must have at least two unique groups for fairness metrics. Found: {self.groups}")

        # If reference_group is specified, ensure it exists
        if self.reference_group is not None and self.reference_group not in self.groups:
            raise ValueError(f"Reference group '{self.reference_group}' not found in unique groups of '{self.protected_attr}'. Found: {self.groups}")

    def _calculate_favorable_rate_per_group(self):
        """Calculates the proportion of favorable outcomes for each group."""
        favorable_rates = {}
        for group in self.groups:
            group_df = self.df[self.df[self.protected_attr] == group]
            if not group_df.empty:
                # Calculate mean, convert to float to handle potential np.nan if subgroup is empty
                rate = (group_df[self.prediction_col] == self.positive_label).mean()
                favorable_rates[group] = float(rate) if pd.notna(rate) else 0.0 # Handle NaN for empty subgroups
            else:
                favorable_rates[group] = 0.0 # No samples for this group

        return favorable_rates

    def statistical_parity_gap(self):
        """
        Calculates the Statistical Parity Gap (SPG) as max(rate) - min(rate) across all groups.
        A positive gap indicates the group with the highest favorable outcome rate.
        """
        favorable_rates = self._calculate_favorable_rate_per_group()
        
        if not favorable_rates: # Should not happen if len(self.groups) >= 2
            return np.nan # Or raise an error

        max_rate = max(favorable_rates.values())
        min_rate = min(favorable_rates.values())

        return float(max_rate - min_rate)

    def disparate_impact_ratio(self):
        """
        Calculates the Disparate Impact Ratio (DI).
        If reference_group is specified, it compares each non-reference group to the reference group.
        Otherwise, it calculates max(rate) / min(rate).
        """
        favorable_rates = self._calculate_favorable_rate_per_group()

        if not favorable_rates or len(favorable_rates) < 2:
            return {} if self.reference_group is not None else np.nan # Return empty dict if not enough groups

        if self.reference_group:
            if self.reference_group not in favorable_rates or favorable_rates[self.reference_group] == 0:
                print(f"Warning: Reference group '{self.reference_group}' has zero or no favorable outcomes. DI ratios involving it will be undefined.")
                # Decide how to handle: skip, return infinity, or NaN
                # For robustness, we'll return NaN for that specific ratio.
                ref_rate = favorable_rates[self.reference_group]
            else:
                ref_rate = favorable_rates[self.reference_group]

            ratios = {}
            for group, rate in favorable_rates.items():
                if group != self.reference_group:
                    if ref_rate > 0:
                        ratios[f"DI({group} vs {self.reference_group})"] = float(rate / ref_rate)
                    else:
                        ratios[f"DI({group} vs {self.reference_group})"] = np.inf if rate > 0 else np.nan
            return ratios
        else:
            # If no reference group, return max/min ratio (less informative for multi-group)
            max_rate = max(favorable_rates.values())
            min_rate = min(favorable_rates.values())
            if min_rate > 0:
                return float(max_rate / min_rate)
            else:
                return np.inf if max_rate > 0 else np.nan # Division by zero scenario

    def report(self):
        """Generates a structured dictionary report of fairness metrics."""
        report_data = {
            "protected_attribute": self.protected_attr,
            "positive_label": self.positive_label,
            "groups_found": list(self.groups),
            "favorable_outcome_rates_per_group": self._calculate_favorable_rate_per_group()
        }

        try:
            report_data["statistical_parity_gap"] = self.statistical_parity_gap()
        except ValueError as e:
            report_data["statistical_parity_gap"] = f"Error: {e}"

        try:
            report_data["disparate_impact_ratio"] = self.disparate_impact_ratio()
        except ValueError as e:
            report_data["disparate_impact_ratio"] = f"Error: {e}"
        
        return report_data