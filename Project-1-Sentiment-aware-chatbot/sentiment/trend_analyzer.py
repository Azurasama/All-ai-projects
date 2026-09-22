class TrendAnalyzer:
    @staticmethod
    def calculate_trend(sentiment_history: list) -> str:
        """
        Calculates the sentiment trend based on the history.
        sentiment_history is a list of sentiment strings (e.g., ['neutral', 'negative', 'negative'])
        Returns: "Improving", "Stable", or "Declining"
        """
        if len(sentiment_history) < 2:
            return "Stable"

        # Map sentiments to numerical values
        sentiment_scores = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }

        # Take up to the last 3 turns
        recent_history = sentiment_history[-3:]
        
        scores = [sentiment_scores.get(s.lower(), 0) for s in recent_history]
        
        # Calculate diffs
        diffs = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        avg_diff = sum(diffs) / len(diffs)
        
        if avg_diff > 0:
            return "Improving"
        elif avg_diff < 0:
            return "Declining"
        else:
            return "Stable"
