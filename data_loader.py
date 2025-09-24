from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

def load_and_vectorize_data():
    dataset = load_dataset("vpmoreira/offcombr", split="train")
    texts = dataset['text']
    labels = dataset['label']

    vectorizer = CountVectorizer(stop_words='portuguese', max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()
    y = labels

    return X, y, vectorizer