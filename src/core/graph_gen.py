import networkx as nx
import matplotlib.pyplot as plt
import io

def extract_graph_data(llm, summary_text):
    """
    Uses Llama 3 to parse a summary into Nodes and Edges.
    """
    prompt = f"""
    Analyze the following research summary. Identify key concepts and their relationships.
    Output ONLY a list of comma-separated tuples in this format:
    (Source Node, Relation, Target Node)
    
    Example:
    (BERT, uses, Transformer Architecture)
    (Llama 3, outperforms, GPT-3.5)

    Summary:
    {summary_text}
    
    Tuples:
    """
    response = llm.invoke(prompt)
    
    # Simple parsing logic (In production, use Regex or JSON mode)
    relationships = []
    for line in response.split('\n'):
        if '(' in line and ')' in line and ',' in line:
            clean_line = line.strip().replace('(', '').replace(')', '')
            parts = clean_line.split(',')
            if len(parts) >= 3:
                relationships.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    
    return relationships

def generate_network_graph(relationships):
    """
    Draws the graph and returns a BytesIO object for downloading.
    """
    G = nx.DiGraph()
    for source, rel, target in relationships:
        G.add_edge(source, target, label=rel)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5)  # k regulates distance between nodes
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=9, font_weight='bold', edge_color='gray')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer