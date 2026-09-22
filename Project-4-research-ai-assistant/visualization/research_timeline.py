import plotly.express as px
import pandas as pd
from collections import defaultdict

class TimelineVisualizer:
    
    @staticmethod
    def create_timeline(papers: list[dict]):
        """
        Creates a plotly figure showing number of papers per year.
        Assumes papers have 'update_date' metadata in 'YYYY-MM-DD' format.
        """
        if not papers:
            return None
            
        # Extract years
        years = []
        for p in papers:
            date_str = p.get('metadata', {}).get('update_date', '')
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    years.append(year)
                except ValueError:
                    pass
                    
        if not years:
            return None
            
        # Count papers per year
        counts = defaultdict(int)
        for y in years:
            counts[y] += 1
            
        df = pd.DataFrame({
            "Year": list(counts.keys()),
            "Papers": list(counts.values())
        })
        
        df = df.sort_values(by="Year")
        
        fig = px.bar(df, x="Year", y="Papers", title="Research Timeline",
                     labels={"Year": "Publication/Update Year", "Papers": "Number of Papers"})
        
        fig.update_layout(xaxis=dict(tickmode='linear'))
        return fig
