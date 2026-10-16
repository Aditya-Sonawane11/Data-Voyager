# DataVoyager

> An offline-first Python EDA library with interactive reports and conversational assistance.

---

## Overview

**DataVoyager** makes exploratory data analysis fast, verifiable, and privacy-preserving. It enables automated dataset profiling, interactive standalone HTML reports, and natural-language dataset exploration without sending your data to external servers.

### Key Features

- **Accurate & Fast**: Strict statistical calculations with Pandas and NumPy.
- **Privacy & Offline First**: Zero telemetry; standalone reports work air-gapped without external CDN dependencies.
- **Interactive Visualizations**: Embedded Plotly histograms, outlier box plots, frequency bars, and correlation heatmaps.
- **Structured Profiles**: Export analysis cleanly to JSON or Python dataclasses.
- **Data Quality Diagnostics**: Outlier detection, duplicate identification, constant/quasi-constant warnings, and missingness audits.
- **Conversational Exploration**: Ask questions in plain English through a verified deterministic assistant without requiring an LLM or API keys.

---

## Installation

```bash
# Editable installation for local development
pip install -e .

# With development tools
pip install -e .[dev]
```

---

## Quickstart

```python
import datavoyager

# Load dataset
df = datavoyager.read_csv("dataset.csv")

# Generate profile
report = datavoyager.analyze(df, title="Customer Analytics Profile")

# Quick terminal summary
report.summary()

# Export profile to JSON
report.to_json("profile.json")

# Generate standalone offline HTML report
report.to_html("report.html")

# Chat with your dataset (deterministic offline assistant)
assistant = report.chat(mode="offline")
answer = assistant.ask("Which columns have the most missing values?")
print(answer)
```

---

## License

Proprietary © 2026 Aditya. All rights reserved.
