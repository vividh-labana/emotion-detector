from functools import lru_cache
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-students"

@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    id2label = model.config.id2label
    return tokenizer, model, id2label

def predict(text: str, top_k: int = 5) -> List[Dict]:
    if not isinstance(text, str) or not text.strip():
        return []
    tokenizer, model, id2label = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    probs_list = probs.tolist()
    indices = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)[:top_k]
    return [{"label": id2label[i], "score": float(probs_list[i])} for i in indices]