import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

############################
# 1) Streamlit Page Config
############################
# Must be called before any other st.* calls
st.set_page_config(
    page_title="4E Politician-Style Classifier",
    page_icon="🦅",
    layout="centered"
)

############################
# 2) Model Definition (Same architecture, smaller dims)
############################
class TweetClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=1, dropout=0.5):
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
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        context_vector = self.layer_norm(context_vector)
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)
        return logits

############################
# 3) Load Model + Tokenizer
############################
@st.cache_resource
def load_model_and_assets():
    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    label_classes = np.load("label_encoder.npy", allow_pickle=True)
    num_classes = len(label_classes)

    # Define the same smaller architecture
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 1
    dropout = 0.5

    model = TweetClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_layers=num_layers,
        dropout=dropout
    )
    # Load weights
    state_dict = torch.load("best_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Optional: Dynamic quantization to reduce memory usage
    # We'll quantize LSTM and Linear layers
    model_quant = quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)

    return model_quant, tokenizer, label_classes

model, tokenizer, label_classes = load_model_and_assets()

############################
# 4) Prediction Function
############################
def predict_tweet(tweet: str, max_length=100):
    seq = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    input_tensor = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)  # shape: [1, num_classes]

    probabilities = probs.squeeze(0).numpy()
    # Convert to percentages
    results = {
        author: float(prob * 100)
        for author, prob in zip(label_classes, probabilities)
    }

    # Sort descending
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1], reverse=True)
    )
    return sorted_results

############################
# 5) Streamlit UI
############################
st.title("🦅 4E Politician-Style Classifier")
st.markdown(
    """
    **A snazzy tool that analyzes your tweet**  
    and guesses which politician's style it matches!  
    ---
    """
)

tweet_input = st.text_area("Enter a tweet below:", height=120)

if st.button("Analyze"):
    txt = tweet_input.strip()
    if not txt:
        st.warning("Please enter something to classify!")
    else:
        results = predict_tweet(txt)
        st.subheader("Style Match Results (in %)")
        for author, pct in results.items():
            st.write(f"- **{author}**: {pct:.2f}%")

st.markdown(
    """
    ---
    **Provided by 4E News**  
    *Because democracy thrives on knowledge!* 🦅🦅🦅🦅
    """
)

