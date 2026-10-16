# Changelog

All notable changes to **Datax** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- Core exploratory data analysis engine (`datax.analyze`, `datax.read_csv`).
- Dataset overview and automatic type inference (numerical, categorical, boolean, datetime, complex).
- Detailed column-level summary statistics, quartiles, skewness, and category frequencies.
- Data quality analytics: missing value rates, duplicate rows, constant/quasi-constant columns, candidate identifier detection, and IQR-based outlier boundaries.
- Correlation analysis supporting Pearson and Spearman with configurable column thresholds.
- Rule-based automated insight discovery engine.
- Structured profile representation (`DatasetProfile`) with JSON serialization and console summaries.
- Comprehensive unit test suite with edge-case fixtures.
