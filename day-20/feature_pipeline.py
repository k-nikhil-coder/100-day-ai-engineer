"""
Day 20: SQL Feature Engineering Pipeline
Extracts clean, ML-ready training features from relational transactional tables
using Common Table Expressions (CTEs) and Window Functions.
"""

import sqlite3
from typing import Optional
import pandas as pd


def initialize_database(db_path: str = "housing_analytics.db") -> None:
    """Sets up transactional tables and seeds mock transaction records."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            sale_price REAL NOT NULL,
            sale_date TEXT NOT NULL,
            sqft INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL
        );

        -- Clear existing demo rows to maintain idempotent runs
        DELETE FROM transactions;

        INSERT INTO transactions (property_id, sale_price, sale_date, sqft, bedrooms) VALUES
        (101, 350000.0, '2025-01-15', 1800, 3),
        (101, 375000.0, '2025-06-20', 1800, 3),
        (102, 520000.0, '2025-02-10', 2400, 4),
        (103, 210000.0, '2025-03-05', 1100, 2),
        (102, 545000.0, '2025-08-12', 2400, 4),
        (104, 430000.0, '2025-04-18', 1950, 3);
        """)
        conn.commit()


def extract_ml_features(db_path: str = "housing_analytics.db") -> pd.DataFrame:
    """
    Executes an analytical SQL query utilizing:
      - Common Table Expression (CTE) for staged transformations
      - Window function (ROW_NUMBER) to isolate the most recent transaction per property
      - Computed features (price_per_sqft) directly at the database layer
    """
    query = """
    WITH ranked_sales AS (
        SELECT 
            property_id,
            sale_price,
            sale_date,
            sqft,
            bedrooms,
            ROUND(sale_price / sqft, 2) AS price_per_sqft,
            ROW_NUMBER() OVER (
                PARTITION BY property_id 
                ORDER BY sale_date DESC
            ) AS recency_rank
        FROM transactions
    )
    SELECT 
        property_id,
        sale_price AS latest_price,
        sqft,
        bedrooms,
        price_per_sqft
    FROM ranked_sales
    WHERE recency_rank = 1
    ORDER BY property_id ASC;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def main() -> None:
    db_file = "housing_analytics.db"

    print("[1/2] Initializing SQLite database and seeding transactions...")
    initialize_database(db_file)

    print("[2/2] Running analytical feature extraction query...")
    df_features = extract_ml_features(db_file)

    print("\n--- Extracted ML Training Features ---")
    print(df_features.to_string(index=False))
    print(f"\nFeature Matrix Shape: {df_features.shape}")


if __name__ == "__main__":
    main()
