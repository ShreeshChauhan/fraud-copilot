# Fraud Investigation Copilot

An AI-powered financial fraud detection and investigation system that combines Graph Neural Networks with Large Language Models to automate the fraud investigation pipeline — reducing manual investigation time from hours to seconds.

## The Problem
Banks receive thousands of fraud alerts daily. Human investigators spend 2–4 hours per case manually reviewing transaction history, mapping account connections, and writing Suspicious Activity Reports (SARs) — a legal regulatory requirement. This project automates that entire pipeline.

## Solution
A 6-layer AI pipeline that takes a flagged account ID and produces a complete investigation dossier in under 60 seconds.
Transaction Data → GNN Fraud Detection → Risk Explanation → Graph Analysis → Investigator Chat → SAR Report

## Features
- **GNN Fraud Detection** — GraphSAGE model that learns fraud patterns from transaction network structure, detecting money rings, structuring, and layering that traditional ML misses
- **AI Risk Explanation** — Gemini LLM generates professional compliance narratives explaining exactly why an account is suspicious
- **Transaction Graph Visualization** — Interactive network map showing money flow and suspicious clusters
- **Investigator Chat** — Analysts query transaction data in plain English and get instant answers
- **SAR Report Generation** — One-click PDF export of the complete investigation dossier

## Tech Stack

| Layer | Technology |
|---|---|
| Graph ML | PyTorch Geometric, GraphSAGE |
| LLM | Gemini API (Google) |
| Data | Python, Pandas, NetworkX |
| Backend | FastAPI |
| Frontend | Streamlit |
| Reports | ReportLab |

## Results
- Achieved 95% accuracy and perfect recall on a synthetic AML transaction dataset generated to simulate money laundering behaviors.
- Investigation time reduced from 2–4 hours to under 60 seconds

## Project Structure
fraud-copilot/

├── data/              # Synthetic transaction data generator

├── graph/             # Transaction graph construction

├── model/             # GNN fraud detection model

├── llm/               # Gemini API explanation + chat layer

├── api/               # FastAPI backend

├── ui/                # Streamlit frontend

└── reports/           # SAR report generator

## Setup

```bash
git clone https://github.com/ShreeshChauhan/fraud-copilot.git
cd fraud-copilot
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Add your API key to `.env`:
GEMINI_API_KEY=your_key_here

Run the app:
```bash
streamlit run ui/app.py
```

## Background
Built as a personal project to explore AI applications in financial compliance. Inspired by the real-world pain point that AML investigators face — banks file 3M+ SARs per year, each requiring hours of manual work.

## Author
Shreesh Chauhan — Computer Science, Stony Brook University
