import json
import evaluate 
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import os

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Load both files
with open("my_conversations.json") as f:
    real_data = json.load(f)

with open("generated_conversations.json") as f:
    gen_data = json.load(f)

# Extract assistant responses from conversations
real_responses = [
    " ".join(msg["value"] for msg in conv["conversations"] if msg["from"] == "assistant")
    for conv in real_data
]

gen_responses = [
    " ".join(msg["value"] for msg in conv["conversations"] if msg["from"] == "assistant")
    for conv in gen_data
]

# Print counts and lengths (word counts)
print(f"Real: {len(real_data)} conversations")
print(f"Generated: {len(gen_data)} conversations")

print("\nWord counts per conversation:")
for i, (real_text, gen_text) in enumerate(zip(real_responses, gen_responses)):
    real_len = len(real_text.split())
    gen_len = len(gen_text.split())
    print(f"Conversation {i+1}: Real words = {real_len}, Generated words = {gen_len}")

# Optionally print total word counts as well
total_real_words = sum(len(text.split()) for text in real_responses)
total_gen_words = sum(len(text.split()) for text in gen_responses)
print(f"\nTotal words - Real: {total_real_words}, Generated: {total_gen_words}")

# Print counts and total lengths
print(f"Real: {len(real_data)} conversations, total length of assistant responses: {sum(len(r) for r in real_responses)} chars")
print(f"Generated: {len(gen_data)} conversations, total length of assistant responses: {sum(len(r) for r in gen_responses)} chars")

# Compare first conversation structures
print("\nExample comparison:")
print("Real:", real_data[0])
print("Generated:", gen_data[0])

# 1) TF-IDF Cosine Similarity
vectorizer = TfidfVectorizer().fit(real_responses + gen_responses)

print("\nTF-IDF Cosine Similarities:")
for i, (real, gen) in enumerate(zip(real_responses, gen_responses)):
    vecs = vectorizer.transform([real, gen])
    sim = cosine_similarity(vecs[0], vecs[1])[0][0]
    print(f"Conversation {i+1} similarity: {sim:.2f}")

# 2) ROUGE Scores
scores = rouge.compute(
    predictions=gen_responses,
    references=real_responses
)
print("\nROUGE Scores:")
print(scores)

# 3) Semantic Similarity with Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & effective model

real_embeddings = model.encode(real_responses, convert_to_tensor=True)
gen_embeddings = model.encode(gen_responses, convert_to_tensor=True)

similarities = util.cos_sim(real_embeddings, gen_embeddings)

print("\nSemantic Similarities (Sentence-BERT):")
for i in range(len(real_responses)):
    sim_score = similarities[i][i].item()
    print(f"Conversation {i+1} semantic similarity: {sim_score:.4f}")

avg_sim = similarities.diagonal().mean().item()
print(f"\nAverage semantic similarity: {avg_sim:.4f}")


from transformers import pipeline


def get_sentiment_scores(texts):
    payload = {
        "samples": texts,
        "threshold": 0.5
    }

    try:
        response = requests.post(
            "https://metrics.twon.uni-trier.de/",
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("predictions", [])
    except Exception as e:
        print("Error calling sentiment API:", e)
        return []

def extract_emotion_scores(item):
    results = item.get("results", {})
    return {
        "hate": results.get("hate", {}).get("HATE", 0),
        "not_hate": results.get("hate", {}).get("NOT-HATE", 0),
        "non_offensive": results.get("offensive", {}).get("non-offensive", 0),
        "irony": results.get("irony", {}).get("irony", 0),
        "neutral": results.get("sentiment", {}).get("neutral", 0),
        "positive": results.get("sentiment", {}).get("positive", 0),
        "negative": results.get("sentiment", {}).get("negative", 0),
    }

# Get sentiment data for real and generated responses
real_sentiments_raw = get_sentiment_scores(real_responses)
gen_sentiments_raw = get_sentiment_scores(gen_responses)

# Parse the raw data to extract flat emotion scores
real_sentiments = [extract_emotion_scores(item) for item in real_sentiments_raw]
gen_sentiments = [extract_emotion_scores(item) for item in gen_sentiments_raw]

# Emotions list (consistent order)
emotions = ["hate", "not_hate", "non_offensive", "irony", "neutral", "positive", "negative"]

# Create output directory for charts
os.makedirs("emotion_charts", exist_ok=True)

num_conversations = min(len(real_sentiments), len(gen_sentiments))

for i in range(num_conversations):
    real_scores = [real_sentiments[i].get(e, 0) for e in emotions]
    gen_scores = [gen_sentiments[i].get(e, 0) for e in emotions]

    x = np.arange(len(emotions))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, real_scores, width, label='Real')
    bars2 = ax.bar(x + width/2, gen_scores, width, label='Generated')

    ax.set_ylabel('Emotion Scores')
    ax.set_title(f'Emotion Comparison for Conversation {i+1}')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45)
    ax.legend()

    # Annotate bars with values
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(f'emotion_charts/conversation_{i+1}_emotion_comparison.png')
    plt.close()

print(f"✅ Saved {num_conversations} individual emotion comparison charts to 'emotion_charts/' folder.")



