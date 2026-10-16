"""Basic end-to-end example using DataVoyager for exploratory data analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

import datavoyager


def main() -> None:
    # 1. Synthesize a realistic sample customer dataset
    np.random.seed(42)
    n = 250
    df = pd.DataFrame({
        "customer_id": range(1001, 1001 + n),
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.normal(65000, 18000, size=n),
        "credit_score": np.random.normal(700, 50, size=n),
        "subscription_tier": np.random.choice(["Basic", "Pro", "Enterprise"], size=n, p=[0.5, 0.35, 0.15]),
        "churned": np.random.choice([False, True], size=n, p=[0.82, 0.18]),
        "account_created": pd.date_range("2023-01-01", periods=n, freq="D"),
    })

    # Add realistic data anomalies
    # Missing values
    df.loc[10:25, "income"] = np.nan
    # An extreme outlier
    df.loc[5, "income"] = 1_250_000.0

    print("Analyzing dataset with DataVoyager...")
    report = datavoyager.analyze(df, title="Customer Churn & Demographics EDA")

    # 2. Print terminal summary
    print("\n" + "=" * 60)
    report.summary()

    # 3. Export profile to JSON
    report.to_json("customer_profile.json")
    print("\nSaved JSON profile to 'customer_profile.json'.")

    # 4. Export standalone offline HTML report
    report.to_html("customer_report.html")
    print("Saved offline HTML report to 'customer_report.html'.")

    # 5. Interrogate dataset using the offline deterministic AI assistant
    assistant = report.chat(mode="offline")
    print("\n" + "=" * 60)
    print("AI Assistant (Offline Deterministic Mode):")
    print("-" * 60)

    q1 = "Which columns have the most missing values?"
    print(f"Q: {q1}")
    print(f"A:\n{assistant.ask(q1)}\n")

    q2 = "Are there any outliers in the dataset?"
    print(f"Q: {q2}")
    print(f"A:\n{assistant.ask(q2)}\n")

    q3 = "Tell me about the income column."
    print(f"Q: {q3}")
    print(f"A:\n{assistant.ask(q3)}\n")


if __name__ == "__main__":
    main()
