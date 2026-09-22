import json

# A small evaluation dataset as requested
DATASET = [
    {
        "text": "Your service is amazing! The issue was solved very quickly.",
        "expected_sentiment": "positive",
        "expected_intent": "feedback"
    },
    {
        "text": "I've been trying to reset my password for hours and nothing works. This is completely unacceptable.",
        "expected_sentiment": "negative",
        "expected_intent": "password_reset"
    },
    {
        "text": "How can I change my email address?",
        "expected_sentiment": "neutral",
        "expected_intent": "account_management"
    },
    {
        "text": "I'm incredibly frustrated. My account is locked and I need to buy a ticket right now!",
        "expected_sentiment": "negative",
        "expected_intent": "account_unlock"
    },
    {
        "text": "Thank you! That fixed the problem immediately. Can I also get a receipt?",
        "expected_sentiment": "positive",
        "expected_intent": "request_receipt"
    },
    {
        "text": "The product is excellent, but delivery was extremely slow.",
        "expected_sentiment": "mixed", # or neutral/negative depending on model
        "expected_intent": "feedback"
    },
    {
        "text": "Great, another error. Exactly what I needed.",
        "expected_sentiment": "negative", # sarcastic
        "expected_intent": "technical_support"
    },
    {
        "text": "I am not happy with the service.",
        "expected_sentiment": "negative",
        "expected_intent": "complaint"
    },
    {
        "text": "That's not a bad solution, thanks.",
        "expected_sentiment": "positive",
        "expected_intent": "feedback"
    },
    {
        "text": "My order hasn't arrived.",
        "expected_sentiment": "neutral",
        "expected_intent": "order_status"
    },
    {
        "text": "I've already waited a week! Help me now!",
        "expected_sentiment": "negative",
        "expected_intent": "order_status"
    }
]

def load_dataset():
    return DATASET
