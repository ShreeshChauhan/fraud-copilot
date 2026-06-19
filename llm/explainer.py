from google import genai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_account_summary(account_id):
    transactions = pd.read_csv("data/transactions.csv")
    accounts     = pd.read_csv("data/accounts.csv")

    account_info = accounts.set_index("account_id").loc[account_id]

    sent = transactions[transactions["sender_id"] == account_id]
    received = transactions[transactions["receiver_id"] == account_id]

    top_sent = sent.nlargest(5, "amount")[["receiver_id", "amount", "tx_type", "timestamp", "flag_reason"]]
    top_received = received.nlargest(5, "amount")[["sender_id", "amount", "tx_type", "timestamp"]]

    summary = {
        "account_id":        account_id,
        "account_type":      account_info["account_type"],
        "country":           account_info["country"],
        "total_sent":        round(sent["amount"].sum(), 2),
        "total_received":    round(received["amount"].sum(), 2),
        "times_sent":        len(sent),
        "times_received":    len(received),
        "largest_sent":      round(sent["amount"].max(), 2) if len(sent) > 0 else 0,
        "fraud_tx_count":    int(sent["is_fraud"].sum()),
        "flag_reasons":      sent["flag_reason"].dropna().unique().tolist(),
        "top_sent":          top_sent.to_dict("records"),
        "top_received":      top_received.to_dict("records"),
        "unique_receivers":  sent["receiver_id"].nunique(),
        "unique_senders":    received["sender_id"].nunique(),
    }
    return summary


def generate_risk_explanation(account_id, risk_score):
    summary = get_account_summary(account_id)

    prompt = f"""You are a senior AML compliance analyst at a US bank.
Analyze the following account data and write a professional 2-paragraph SAR narrative finding.
Be specific about amounts, dates, patterns, and flag reasons.
Use formal regulatory language. Never speculate beyond the data provided.

Account ID: {account_id}
Account Type: {summary['account_type']}
Country: {summary['country']}
Risk Score: {risk_score:.2f} out of 1.0

Transaction Activity:
- Total sent: ${summary['total_sent']:,}
- Total received: ${summary['total_received']:,}
- Number of outgoing transactions: {summary['times_sent']}
- Number of incoming transactions: {summary['times_received']}
- Largest single outgoing amount: ${summary['largest_sent']:,}
- Unique counterparties sent to: {summary['unique_receivers']}
- Flagged transactions: {summary['fraud_tx_count']}
- Flag reasons detected: {', '.join(summary['flag_reasons']) if summary['flag_reasons'] else 'None'}

Top outgoing transactions:
{pd.DataFrame(summary['top_sent']).to_string() if summary['top_sent'] else 'None'}

Write the SAR narrative finding now:"""

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
    )
    return response.text


def answer_investigator_question(account_id, question, risk_score):
    summary = get_account_summary(account_id)

    prompt = f"""You are an AI assistant helping a fraud investigator at a bank.
You have access to the full case file for account {account_id}.
Answer the investigator's question using only the data provided.
Be concise and specific.

Case File:
- Account Type: {summary['account_type']}
- Country: {summary['country']}
- Risk Score: {risk_score:.2f}
- Total sent: ${summary['total_sent']:,}
- Total received: ${summary['total_received']:,}
- Flagged transactions: {summary['fraud_tx_count']}
- Flag reasons: {', '.join(summary['flag_reasons']) if summary['flag_reasons'] else 'None'}
- Unique counterparties: {summary['unique_receivers']} sent to, {summary['unique_senders']} received from
- Top transactions: {summary['top_sent']}

Investigator question: {question}

Answer:"""

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
    )
    return response.text


if __name__ == "__main__":
    accounts = pd.read_csv("data/accounts.csv")
    fraud_accounts = accounts[accounts["is_fraud"] == 1]["account_id"].tolist()

    if fraud_accounts:
        test_account = fraud_accounts[0]
        print(f"Testing with fraud account: {test_account}")
        print("\nGenerating risk explanation...\n")
        explanation = generate_risk_explanation(test_account, risk_score=0.91)
        print(explanation)
        print("\n--- Investigator Chat Test ---")
        answer = answer_investigator_question(
            test_account,
            "What are the most suspicious transactions for this account?",
            risk_score=0.91
        )
        print(answer)

    