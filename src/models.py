import streamlit as st

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@st.cache_resource(show_spinner=False)
def load_roberta():
    """Load RoBERTa sentiment model. Returns (tokenizer, model, error_str)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def load_vader():
    """Load VADER analyzer. Returns (analyzer, error_str)."""
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer(), None
    except Exception as e:
        return None, str(e)


def predict_roberta(text: str, tokenizer, model) -> dict:
    """Run RoBERTa inference on text."""
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()

    id2label = model.config.id2label
    scores = {'negative': 0.0, 'neutral': 0.0, 'positive': 0.0}
    for idx, prob in enumerate(probs):
        name = id2label[idx].lower()
        if 'neg' in name:
            scores['negative'] = prob
        elif 'neu' in name:
            scores['neutral'] = prob
        elif 'pos' in name:
            scores['positive'] = prob

    pred = max(scores, key=scores.get)
    return {**scores, 'label': pred, 'confidence': scores[pred]}


def predict_vader(text: str, analyzer) -> dict:
    """Run VADER inference on text."""
    s = analyzer.polarity_scores(text)
    compound = s['compound']
    label = (
        'positive' if compound >= 0.05
        else 'negative' if compound <= -0.05
        else 'neutral'
    )
    return {
        'negative': s['neg'],
        'neutral':  s['neu'],
        'positive': s['pos'],
        'compound': compound,
        'label': label,
    }
