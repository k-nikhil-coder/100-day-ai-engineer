# Day 20: SQL for Data Engineering & Feature Extraction

An analytical data pipeline that shifts feature engineering from in-memory processing to relational database queries using SQLite, Pandas, Common Table Expressions (CTEs), and Window Functions.

---

## 📌 Technical Overview

In production machine learning systems, feature datasets are rarely loaded from static flat CSVs. They are extracted directly from relational data stores and warehouses. 

This pipeline demonstrates:
* **Window Functions (`ROW_NUMBER() OVER (...)`):** Partitions raw transaction logs by `property_id` and orders by `sale_date DESC` to isolate the most recent valuation without data leakage.
* **Common Table Expressions (CTEs):** Encapsulates preliminary ranking and derived computations in a readable, reusable `WITH` block.
* **Database-Level Feature Derivation:** Computes `price_per_sqft` in the SQL engine before loading results into a Pandas DataFrame, minimizing in-memory Python overhead.

---

## 📂 Directory Structure

```text
day-20/
├── feature_pipeline.py    # Database initialization and feature extraction pipeline
├── requirements.txt       # Dependencies
└── README.md              # Pipeline documentation
