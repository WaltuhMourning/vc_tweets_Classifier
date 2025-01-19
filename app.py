import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

############################
# 1) STREAMLIT PAGE CONFIG
############################
# Must be called before any other st.* command
st.set_page_config(
    page_title="4E Politician-Style Classifier",
    page_icon="🦅",
    layout="centered"
)

############################
# 2) MODEL DEFINITION
############################
class TweetClassifier(nn.Module):
    """
    Matches the attention-based architecture from train.py.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.6):
        super(TweetClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        # Weighted sum (context vector)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        context_vector = self.layer_norm(context_vector)
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)
        return logits

############################
# 3) LOAD MODEL / TOKENIZER
############################
@st.cache_resource  # Ensures these are loaded only once
def load_model_and_assets():
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load label encoder classes
    label_classes = np.load('label_encoder.npy', allow_pickle=True)
    num_classes = len(label_classes)

    # Recreate the same model architecture
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 256
    hidden_dim = 512
    model = TweetClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_layers=2,
        dropout=0.6
    )

    # Load the saved state_dict
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer, label_classes

model, tokenizer, label_classes = load_model_and_assets()

############################
# 4) INFERENCE FUNCTION
############################
def predict_tweet(tweet_text: str, max_length=100):
    """
    Convert raw text to probability distribution over authors.
    Returns a dict {author: probability_in_percent}, sorted desc.
    """
    seq = tokenizer.texts_to_sequences([tweet_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    input_tensor = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)         # shape: [1, num_classes]
        probs = F.softmax(logits, dim=1)     # convert to probabilities
    probabilities = probs.squeeze(0).numpy()  # [num_classes]

    # Turn float probabilities into percentages
    results = {
        author: float(prob * 100)
        for author, prob in zip(label_classes, probabilities)
    }

    # Sort in descending order
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return sorted_results

############################
# 5) STREAMLIT UI LAYOUT
############################
st.title("🦅 4E Politician-Style Classifier")
st.markdown(
    """
    ##### A snazzy tool that analyzes your tweet and guesses **which politician’s style** it matches!  
    ---
    """
)

tweet_input = st.text_area("Enter a tweet here:", height=120)

if st.button("Analyze"):
    tweet_text = tweet_input.strip()
    if not tweet_text:
        st.warning("Please enter some text to classify!")
    else:
        results_dict = predict_tweet(tweet_text)
        st.subheader("Style Match Results (in %)")
        for author, pct in results_dict.items():
            st.write(f"- **{author}**: {pct:.2f}%")

st.markdown(
    """
    ---
    **Powered by 4E News** | *Because democracy thrives on knowledge!* 🦅🦅🦅🦅
    """
)

