import pandas as pd
import networkx as nx
from pyvis.network import Network

def build_account_network(account_id, depth=2):
    transactions = pd.read_csv("data/transactions.csv")
    accounts     = pd.read_csv("data/accounts.csv")

    G = nx.DiGraph()
    visited = {account_id}
    frontier = {account_id}

    # Expand the network outward by `depth` hops
    for _ in range(depth):
        next_frontier = set()
        related = transactions[
            (transactions["sender_id"].isin(frontier)) |
            (transactions["receiver_id"].isin(frontier))
        ]
        for _, tx in related.iterrows():
            G.add_edge(tx["sender_id"], tx["receiver_id"],
                       amount=tx["amount"], is_fraud=tx["is_fraud"])
            next_frontier.add(tx["sender_id"])
            next_frontier.add(tx["receiver_id"])
        visited |= next_frontier
        frontier = next_frontier - visited

    return G, visited


def render_network_html(account_id, output_path="ui/network.html"):
    G, visited = build_account_network(account_id)
    accounts = pd.read_csv("data/accounts.csv").set_index("account_id")

    net = Network(height="500px", width="100%", directed=True, bgcolor="#1a1a1a", font_color="white")

    for node in G.nodes():
        is_fraud = accounts.loc[node, "is_fraud"] if node in accounts.index else 0
        is_target = node == account_id

        if is_target:
            color = "#FFD700"   # gold - the searched account
            size = 30
        elif is_fraud:
            color = "#FF4444"   # red - known fraud
            size = 20
        else:
            color = "#4488FF"   # blue - normal
            size = 15

        net.add_node(node, label=node[-8:], color=color, size=size, title=node)

    for source, target, data in G.edges(data=True):
        edge_color = "#FF4444" if data.get("is_fraud") else "#888888"
        net.add_edge(source, target, value=data["amount"]/1000, color=edge_color,
                    title=f"${data['amount']:,.2f}")

    net.set_options("""
    {
      "physics": { "barnesHut": { "gravitationalConstant": -3000, "springLength": 150 } }
    }
    """)
    net.save_graph(output_path)
    return output_path


if __name__ == "__main__":
    accounts = pd.read_csv("data/accounts.csv")
    fraud_accounts = accounts[accounts["is_fraud"] == 1]["account_id"].tolist()

    if fraud_accounts:
        test_account = fraud_accounts[0]
        print(f"Building network for: {test_account}")
        path = render_network_html(test_account)
        print(f"Network saved to {path}")