import streamlit as st
import requests
import pandas as pd
import os

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fraud Investigation Copilot",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fraud Investigation Copilot")
st.caption("AI-powered financial fraud detection and investigation system")

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("Search Account")
    account_id = st.text_input("Enter Account ID", placeholder="ACC-XXXX-XXXX")
    search_btn = st.button("Investigate", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Quick Select")
    st.caption("Known fraud accounts:")

    try:
        accounts = pd.read_csv("data/accounts.csv")
        fraud_accounts = accounts[accounts["is_fraud"] == 1]["account_id"].tolist()[:5]
        for acc in fraud_accounts:
            if st.button(acc, use_container_width=True):
                account_id = acc
                search_btn = True
    except:
        st.warning("Could not load accounts")

# ── Main content ─────────────────────────────────────
if search_btn and account_id:
    account_id = account_id.strip()

    # Risk score
    with st.spinner("Running GNN fraud detection..."):
        risk_response = requests.get(f"{API_URL}/account/{account_id}/risk")

    if risk_response.status_code != 200 or "error" in risk_response.json():
        st.error(f"Account {account_id} not found.")
        st.stop()

    risk_data  = risk_response.json()
    risk_score = risk_data["risk_score"]

    # ── Top metrics row ───────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Account ID", account_id)
    with col2:
        risk_pct = f"{risk_score * 100:.1f}%"
        st.metric("Risk Score", risk_pct)
    with col3:
        if risk_score >= 0.7:
            st.metric("Risk Level", "🔴 HIGH")
        elif risk_score >= 0.4:
            st.metric("Risk Level", "🟡 MEDIUM")
        else:
            st.metric("Risk Level", "🟢 LOW")
    with col4:
        summary_resp = requests.get(f"{API_URL}/account/{account_id}/summary")
        summary = summary_resp.json()
        st.metric("Total Transactions", summary.get("total_transactions", "N/A"))

    st.divider()

    # ── Two column layout ─────────────────────────────
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📊 Account Summary")
        summary_data = {
            "Total Sent":        f"${summary.get('total_sent', 0):,.2f}",
            "Total Received":    f"${summary.get('total_received', 0):,.2f}",
            "Times Sent":        summary.get("times_sent", 0),
            "Times Received":    summary.get("times_received", 0),
            "Largest Amount":    f"${summary.get('largest_sent', 0):,.2f}",
            "Flagged Tx Count":  summary.get("fraud_tx_count", 0),
            "Flag Reasons":      ", ".join(summary.get("flag_reasons", [])),
            "Unique Receivers":  summary.get("unique_receivers", 0),
        }
        for key, value in summary_data.items():
            col_a, col_b = st.columns([1, 1])
            col_a.write(f"**{key}**")
            col_b.write(value)

        st.subheader("🕸️ Transaction Network")
        with st.spinner("Building network graph..."):
            net_resp = requests.get(f"{API_URL}/account/{account_id}/network")
        net_path = f"ui/network_{account_id}.html"
        if os.path.exists(net_path):
            with open(net_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=400, scrolling=False)

    with right_col:
        st.subheader("🤖 AI Risk Explanation")
        with st.spinner("Generating SAR narrative..."):
            exp_resp    = requests.get(f"{API_URL}/account/{account_id}/explanation")
            explanation = exp_resp.json().get("explanation", "Could not generate explanation.")
        st.write(explanation)

        st.divider()
        st.subheader("💬 Investigator Chat")
        st.caption("Ask questions about this account")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask something about this account...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                chat_resp = requests.post(
                    f"{API_URL}/account/{account_id}/chat",
                    params={"question": question}
                )
                answer = chat_resp.json().get("answer", "Could not get answer.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

else:
    st.info("Enter an account ID in the sidebar and click Investigate to begin.")
    st.image("https://img.icons8.com/fluency/96/fraud.png", width=80)
    st.markdown("""
    ### How to use
    1. Enter an account ID in the sidebar
    2. Click **Investigate**
    3. View the risk score, AI explanation, and transaction network
    4. Chat with the AI investigator to ask questions about the account
    """)