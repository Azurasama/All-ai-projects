import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentiment.analyzer import SentimentAnalyzer
from evaluation.dataset import load_dataset

def run_evaluation():
    print("Loading Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    dataset = load_dataset()
    
    y_true = []
    y_pred = []
    
    print("\nRunning Evaluation...\n")
    for item in dataset:
        text = item["text"]
        expected_sentiment = item["expected_sentiment"]
        
        # Twitter-roberta doesn't have 'mixed', so we map 'mixed' to whatever it outputs (often negative or neutral)
        # But for evaluation, if it expects mixed and gets positive/negative, it will count as an error unless we handle it.
        # To keep it simple, we'll just track raw output.
        
        result = analyzer.analyze(text)
        predicted_sentiment = result["sentiment"]
        
        y_true.append(expected_sentiment)
        y_pred.append(predicted_sentiment)
        
        print(f"Text: {text}")
        print(f"Expected: {expected_sentiment} | Predicted: {predicted_sentiment}")
        print("-" * 50)
        
    print("\n--- Metrics ---")
    
    labels = list(set(y_true + y_pred))
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(df_cm)

if __name__ == "__main__":
    run_evaluation()
