import networkx as nx
from pyvis.network import Network
import tempfile
import os

class ConceptGraphVisualizer:
    
    @staticmethod
    def generate_html(entities_dict: dict, root_topic: str = "Research Topic") -> str:
        """
        Takes the dictionary of extracted concepts (algorithms, models, datasets, metrics)
        and returns the HTML string for a PyVis graph.
        """
        G = nx.Graph()
        
        # Add root node
        G.add_node(root_topic, group="Root", title="Main Topic", size=25, color="#ff5722")
        
        colors = {
            "algorithms": "#03a9f4",
            "models": "#4caf50",
            "datasets": "#9c27b0",
            "metrics": "#ff9800"
        }
        
        for category, items in entities_dict.items():
            if isinstance(items, list):
                # Add category node
                cat_node = category.capitalize()
                G.add_node(cat_node, group="Category", color="#607d8b", size=20)
                G.add_edge(root_topic, cat_node)
                
                # Add item nodes
                for item in items:
                    if not item: continue
                    G.add_node(item, group=category, color=colors.get(category, "#9e9e9e"), size=15)
                    G.add_edge(cat_node, item)
                    
        # Generate PyVis network
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        
        # Physics config for nice layout
        net.toggle_physics(True)
        
        # Save to temp file and read HTML
        fd, path = tempfile.mkstemp(suffix=".html")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                pass # close immediately
            net.save_graph(path)
            with open(path, 'r', encoding='utf-8') as f:
                html_data = f.read()
        finally:
            os.remove(path)
            
        return html_data
