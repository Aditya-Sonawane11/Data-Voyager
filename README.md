# Datax

> An offline-first Python EDA library with interactive reports and conversational AI assistance.

[![PyPI version](https://img.shields.io/pypi/v/datax-eda.svg)](https://pypi.org/project/datax-eda/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

**Datax** makes exploratory data analysis fast, verifiable, and privacy-preserving. It enables automated dataset profiling, interactive standalone HTML reports, and natural-language dataset exploration without sending your data to external servers by default.

### Key Features

- **Accurate & Fast**: Strict statistical calculations with Pandas and NumPy.
- **Privacy & Offline First**: Zero telemetry; reports work air-gapped without external CDN dependencies.
- **Structured Profiles**: Export analysis cleanly to JSON or Python dataclasses.
- **Data Quality Diagnostics**: Outlier detection, duplicate identification, constant/quasi-constant warnings, and missingness audits.
- **Conversational Exploration**: Ask questions in plain English through a deterministic rule-based assistant or local Ollama LLMs.

---

## Installation

```bash
# Core library
pip install datax-eda

# With local LLM assistance (Ollama adapter)
pip install datax-eda[ai-local]

# Development dependencies
pip install datax-eda[dev]
```

---

## Quickstart

```python
import datax

# Load dataset
df = datax.read_csv("dataset.csv")

# Generate profile
report = datax.analyze(df, title="Customer Analytics Profile")

# Quick terminal summary
report.summary()

# Export profile to JSON
report.to_json("profile.json")

# Generate standalone offline HTML report (Week 2)
# report.to_html("report.html")

# Chat with your dataset (deterministic offline assistant)
assistant = report.chat(mode="offline")
answer = assistant.ask("Which columns have the most missing values?")
print(answer)
```

---

## License

MIT © [Datax Developers](LICENSE)
