"""
Dead Stock Detector
-------------------
Finds dead and slow-moving inventory items from stock and sales data.

Usage:
    python dead_stock_detector.py --demo
    python dead_stock_detector.py --inventory inventory.csv --sales sales.csv

Input files:
    inventory.csv -> product_id, product_name, category, stock_qty, unit_cost
    sales.csv     -> product_id, sale_date (YYYY-MM-DD), qty_sold

Rules (change the thresholds below to suit your business):
    DEAD   : no sales in the last DEAD_DAYS days (or never sold) and stock > 0
    SLOW   : has sales, but days of stock cover > SLOW_COVER_DAYS
    ACTIVE : everything else
"""

import argparse
import numpy as np
import pandas as pd

# ---------------- Settings ----------------
DEAD_DAYS = 180         # no sale in this many days -> dead stock
SLOW_COVER_DAYS = 120   # stock would last longer than this -> slow moving
LOOKBACK_DAYS = 90      # window used to calculate average daily sales


# ---------------- Demo data ----------------
def create_demo_data(n_products=60, seed=42):
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize()
    categories = ["Electronics", "Grocery", "Clothing", "Toys", "Home"]

    inventory = pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(1, n_products + 1)],
        "product_name": [f"Product {i}" for i in range(1, n_products + 1)],
        "category": rng.choice(categories, n_products),
        "stock_qty": rng.integers(0, 300, n_products),
        "unit_cost": rng.integers(5, 500, n_products),
    })

    rows = []
    for pid in inventory["product_id"]:
        kind = rng.choice(["active", "slow", "dead"], p=[0.6, 0.2, 0.2])
        if kind == "active":
            n_sales, window = rng.integers(40, 120), 90
        elif kind == "slow":
            n_sales, window = rng.integers(3, 10), 365
        else:
            n_sales, window = rng.integers(0, 5), 365
        for _ in range(n_sales):
            if kind == "dead":
                days_ago = rng.integers(200, 700)   # only old sales
            else:
                days_ago = rng.integers(0, window)
            rows.append({
                "product_id": pid,
                "sale_date": today - pd.Timedelta(days=int(days_ago)),
                "qty_sold": int(rng.integers(1, 5)),
            })
    sales = pd.DataFrame(rows)
    return inventory, sales


# ---------------- Core logic ----------------
def detect_dead_stock(inventory, sales, today=None):
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    sales = sales.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])

    # Last sale date per product
    last_sale = sales.groupby("product_id")["sale_date"].max().rename("last_sale_date")

    # Units sold in the lookback window
    recent = sales[sales["sale_date"] >= today - pd.Timedelta(days=LOOKBACK_DAYS)]
    recent_units = recent.groupby("product_id")["qty_sold"].sum().rename("units_recent")

    df = (inventory
          .merge(last_sale, on="product_id", how="left")
          .merge(recent_units, on="product_id", how="left"))
    df["units_recent"] = df["units_recent"].fillna(0)

    df["days_since_last_sale"] = (today - df["last_sale_date"]).dt.days
    df["avg_daily_sales"] = df["units_recent"] / LOOKBACK_DAYS
    df["days_of_cover"] = np.where(
        df["avg_daily_sales"] > 0,
        df["stock_qty"] / df["avg_daily_sales"],
        np.inf,
    )
    df["stock_value"] = df["stock_qty"] * df["unit_cost"]

    def classify(row):
        if row["stock_qty"] <= 0:
            return "OUT OF STOCK"
        never_sold = pd.isna(row["days_since_last_sale"])
        if never_sold or row["days_since_last_sale"] > DEAD_DAYS:
            return "DEAD"
        if row["days_of_cover"] > SLOW_COVER_DAYS:
            return "SLOW"
        return "ACTIVE"

    df["status"] = df.apply(classify, axis=1)

    suggestions = {
        "DEAD": "Clearance sale / bundle / return to supplier",
        "SLOW": "Run discount or promotion, stop reordering",
        "ACTIVE": "No action needed",
        "OUT OF STOCK": "Check demand before restocking",
    }
    df["suggestion"] = df["status"].map(suggestions)
    return df


# ---------------- Reporting ----------------
def print_summary(df):
    total_value = df["stock_value"].sum()
    summary = (df.groupby("status")
                 .agg(products=("product_id", "count"),
                      stock_value=("stock_value", "sum"))
                 .reset_index())
    summary["value_share_%"] = (summary["stock_value"] / total_value * 100).round(1)

    print("\n===== DEAD STOCK DETECTOR - SUMMARY =====")
    print(summary.to_string(index=False))

    dead = df[df["status"] == "DEAD"].sort_values("stock_value", ascending=False)
    print(f"\nTop 10 dead stock items by value (total dead value: {dead['stock_value'].sum():,.0f})")
    cols = ["product_id", "product_name", "category", "stock_qty",
            "days_since_last_sale", "stock_value"]
    print(dead[cols].head(10).to_string(index=False))

    print("\nDead stock value by category")
    print(dead.groupby("category")["stock_value"].sum()
              .sort_values(ascending=False).to_string())


def save_chart(df, path="stock_status.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping chart.")
        return
    value = df.groupby("status")["stock_value"].sum()
    ax = value.plot(kind="bar", color="steelblue", figsize=(7, 4))
    ax.set_title("Stock Value by Status")
    ax.set_ylabel("Stock value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Chart saved to {path}")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Dead Stock Detector")
    parser.add_argument("--inventory", help="Path to inventory CSV")
    parser.add_argument("--sales", help="Path to sales CSV")
    parser.add_argument("--demo", action="store_true", help="Use generated demo data")
    parser.add_argument("--out", default="dead_stock_report.csv", help="Output CSV path")
    parser.add_argument("--chart", action="store_true", help="Also save a bar chart")
    args = parser.parse_args()

    if args.demo or not (args.inventory and args.sales):
        print("Using demo data...")
        inventory, sales = create_demo_data()
        inventory.to_csv("inventory.csv", index=False)
        sales.to_csv("sales.csv", index=False)
    else:
        inventory = pd.read_csv(args.inventory)
        sales = pd.read_csv(args.sales)

    result = detect_dead_stock(inventory, sales)
    print_summary(result)

    result.replace([np.inf], np.nan).to_csv(args.out, index=False)
    print(f"\nFull report saved to {args.out}")

    if args.chart:
        save_chart(result)


if __name__ == "__main__":
    main()
