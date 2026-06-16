import pandas as pd
import numpy as np
from faker import Faker
import random
import json
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────
NUM_ACCOUNTS     = 200
NUM_TRANSACTIONS = 2000
FRAUD_RATIO      = 0.08   # 8% of accounts are fraudsters
# ────────────────────────────────────────────────────

def generate_accounts(n):
    accounts = []
    for _ in range(n):
        accounts.append({
            "account_id":   fake.bothify("ACC-####-????").upper(),
            "name":         fake.name(),
            "account_type": random.choice(["personal", "business", "offshore"]),
            "country":      random.choice(["US", "US", "US", "UK", "SG", "AE", "PA"]),
            "join_date":    fake.date_between(start_date="-5y", end_date="-6m").isoformat(),
            "is_fraud":     1 if random.random() < FRAUD_RATIO else 0
        })
    return pd.DataFrame(accounts)


def generate_transactions(accounts_df, n):
    account_ids   = accounts_df["account_id"].tolist()
    fraud_ids     = accounts_df[accounts_df["is_fraud"] == 1]["account_id"].tolist()
    normal_ids    = accounts_df[accounts_df["is_fraud"] == 0]["account_id"].tolist()

    transactions  = []
    start_date    = datetime.now() - timedelta(days=180)

    for i in range(n):
        ts          = start_date + timedelta(
                        days=random.randint(0, 180),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59)
                      )
        is_fraud_tx = random.random() < FRAUD_RATIO

        if is_fraud_tx and fraud_ids:
            sender   = random.choice(fraud_ids)
            # Fraudsters send to each other OR to normal accounts (layering)
            receiver = random.choice(fraud_ids + normal_ids[:20])
        else:
            sender   = random.choice(normal_ids)
            receiver = random.choice(account_ids)

        # Avoid self-transactions
        if sender == receiver:
            receiver = random.choice([a for a in account_ids if a != sender])

        # Fraud transactions: unusual amounts + odd hours
        if is_fraud_tx:
            amount = round(random.choice([
                random.uniform(9000, 9999),    # just under reporting threshold
                random.uniform(500, 2000),     # smurfing (small splits)
                random.uniform(50000, 200000)  # large layering transfer
            ]), 2)
            hour = ts.hour
            # Fraud happens more at night
            if random.random() < 0.6:
                ts = ts.replace(hour=random.randint(1, 5))
        else:
            amount = round(random.uniform(50, 15000), 2)

        transactions.append({
            "transaction_id": fake.bothify("TXN-########").upper(),
            "timestamp":      ts.isoformat(),
            "sender_id":      sender,
            "receiver_id":    receiver,
            "amount":         amount,
            "currency":       "USD",
            "tx_type":        random.choice(["wire", "ach", "internal", "swift"]),
            "is_fraud":       1 if is_fraud_tx else 0,
            "flag_reason":    get_flag_reason(amount, ts.hour) if is_fraud_tx else ""
        })

    return pd.DataFrame(transactions)


def get_flag_reason(amount, hour):
    reasons = []
    if 9000 <= amount <= 9999:
        reasons.append("structuring")
    if amount > 50000:
        reasons.append("large_transfer")
    if 1 <= hour <= 5:
        reasons.append("off_hours")
    if amount < 2000:
        reasons.append("smurfing")
    return "|".join(reasons) if reasons else "suspicious_pattern"


if __name__ == "__main__":
    print("Generating accounts...")
    accounts_df = generate_accounts(NUM_ACCOUNTS)

    print("Generating transactions...")
    transactions_df = generate_transactions(accounts_df, NUM_TRANSACTIONS)

    # Save to CSV
    accounts_df.to_csv("data/accounts.csv", index=False)
    transactions_df.to_csv("data/transactions.csv", index=False)

    # Summary
    print(f"\n✓ Accounts generated:     {len(accounts_df)}")
    print(f"✓ Transactions generated:  {len(transactions_df)}")
    print(f"✓ Fraud accounts:          {accounts_df['is_fraud'].sum()}")
    print(f"✓ Fraud transactions:      {transactions_df['is_fraud'].sum()}")
    print(f"\nSample fraud transaction:")
    print(transactions_df[transactions_df['is_fraud']==1][['transaction_id','amount','flag_reason']].head(3).to_string())
    print("\nData saved to data/accounts.csv and data/transactions.csv")