import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model():
    """Load trained model from HuggingFace Hub"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("CodexTanishq/sentiment-distilbert")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment of input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = outputs.logits.argmax(dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, round(float(confidence), 4)