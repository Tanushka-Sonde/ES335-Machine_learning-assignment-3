# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random, time, requests, re, os, json
from collections import Counter

# -------------------------------------------------------
# ⚙️ Page Config
# -------------------------------------------------------
st.set_page_config(page_title="Next Word Predictor (MLP)", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🔮 Next Word Predictor — MLP Edition")
st.markdown("""
Train and experiment with a **Next-Word Prediction MLP** model on Sherlock Holmes text.

Adjust **context length**, **embedding size**, **layers**, and **activation** —  
then observe how predictions change for your prompt.
""")

# -------------------------------------------------------
# 🧠 Model Definition
# -------------------------------------------------------
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=512, block_size=5, num_layers=2, activation='relu'):
        super().__init__()
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        layers = []
        input_size = block_size * embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(act)
            layers.append(nn.Dropout(0.2))
            input_size = hidden_dim
        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embeds = self.embeddings(x)
        embeds = embeds.view(embeds.size(0), -1)
        return self.mlp(embeds)

# -------------------------------------------------------
# 📚 Dataset
# -------------------------------------------------------
@st.cache_resource
def load_and_preprocess_text():
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    text = requests.get(url).text
    start_marker = "ADVENTURE I. A SCANDAL IN BOHEMIA"
    end_marker = "End of the Project Gutenberg EBook"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = re.sub(r'[^a-zA-Z0-9\s.,;!?\'"-]', '', line).lower().strip()
        if line:
            processed_lines.append(line)

    word_counter = Counter()
    for line in processed_lines:
        word_counter.update(line.split())

    vocab = [w for w, c in word_counter.items() if c >= 2]
    stoi = {w: i+1 for i, w in enumerate(vocab)}
    itos = {i+1: w for i, w in enumerate(vocab)}
    stoi['<UNK>'] = 0
    itos[0] = '<UNK>'
    return processed_lines, stoi, itos

def create_training_pairs(lines, stoi, block_size=5):
    X, Y = [], []
    for line in lines:
        words = line.split()
        if len(words) < 2:
            continue
        context = [0] * block_size
        for w in words + ['.']:
            ix = stoi.get(w, 0)
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# -------------------------------------------------------
# 🏋️ Training
# -------------------------------------------------------
def train_model(model, lines, stoi, epochs=3, lr=0.001):
    X, Y = create_training_pairs(lines, stoi, model.block_size)
    X, Y = X.to(device), Y.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    return model

# -------------------------------------------------------
# 🔮 Prediction
# -------------------------------------------------------
def encode_text(text, stoi, context_len):
    words = text.lower().split()
    encoded = [stoi.get(w, 0) for w in words[-context_len:]]
    while len(encoded) < context_len:
        encoded.insert(0, 0)
    return torch.tensor([encoded], dtype=torch.long).to(device)

def predict_next_words(model, text, k, temperature, stoi, itos, context_len):
    model.eval()
    generated = text.split()
    for _ in range(k):
        x = encode_text(" ".join(generated), stoi, context_len)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(itos.get(next_idx, "<UNK>"))
    return " ".join(generated)

# -------------------------------------------------------
# 🎛️ Sidebar Controls
# -------------------------------------------------------
st.sidebar.header("⚙️ Model Controls")
mode = st.sidebar.radio("Mode", ["Train New Model"])

context_len = st.sidebar.slider("Context Length", 2, 10, 5)
embed_dim = st.sidebar.slider("Embedding Dim", 16, 256, 64, step=16)
hidden_dim = st.sidebar.slider("Hidden Dim", 256, 1024, 512, step=128)
activation = st.sidebar.selectbox("Activation", ["relu", "tanh"])
layers = st.sidebar.slider("Layers", 1, 3, 2)
temperature = st.sidebar.slider("Temperature", 0.2, 2.0, 1.0, 0.1)
k = st.sidebar.slider("Words to Generate", 1, 20, 5)

# 🔁 Epoch Settings (new section)
st.sidebar.markdown("### 🧮 Training Epochs Preset")
epoch_option = st.sidebar.selectbox(
    "Select Epoch Setting",
    ["Low (1)", "Optimum (3)", "High (5)"],
    index=1
)
epoch_map = {"Low (1)": 1, "Optimum (3)": 3, "High (5)": 5}
epochs = epoch_map[epoch_option]

seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

# -------------------------------------------------------
# 📦 Load Data
# -------------------------------------------------------
with st.spinner("📖 Loading Sherlock Holmes..."):
    lines, stoi, itos = load_and_preprocess_text()
vocab_size = len(stoi)

# -------------------------------------------------------
# 💾 Robust Checkpoint Loader
# -------------------------------------------------------
@st.cache_resource
def load_checkpoint(model_path, vocab_size, embed_dim, activation, context_len):
    try:
        state_dict = torch.load(model_path, map_location=device)

        # Infer actual model dimensions from the checkpoint
        def infer_dims(state_dict):
            inferred = {"embed_dim": embed_dim, "hidden_dim": 512, "context_len": context_len}
            for k, v in state_dict.items():
                if "embeddings.weight" in k:
                    inferred["embed_dim"] = v.shape[1]
                elif "mlp.0.weight" in k or "net.0.weight" in k:
                    inferred["hidden_dim"] = v.shape[0]
                    inferred["context_len"] = max(1, v.shape[1] // inferred["embed_dim"])
            return inferred

        inferred = infer_dims(state_dict)

        model = NextWordMLP(
            vocab_size=vocab_size,
            embedding_dim=inferred["embed_dim"],
            hidden_dim=inferred["hidden_dim"],
            activation=activation,
            block_size=inferred["context_len"],
            num_layers=layers
        ).to(device)

        model.load_state_dict(state_dict, strict=False)
        st.success(
            f"✅ Loaded checkpoint successfully — embed_dim={inferred['embed_dim']}, "
            f"context_len={inferred['context_len']}, hidden_dim={inferred['hidden_dim']}"
        )

    except FileNotFoundError:
        st.error(f"❌ Checkpoint not found at {model_path}")
        model = NextWordMLP(vocab_size, embed_dim, 512, activation, context_len, layers).to(device)

    except Exception as e:
        st.warning(f"⚠️ Error loading checkpoint: {e}")
        model = NextWordMLP(vocab_size, embed_dim, 512, activation, context_len, layers).to(device)

    model.eval()
    return model

# -------------------------------------------------------
# 🚀 Mode 1: Train New Model
# -------------------------------------------------------
if mode == "Train New Model":
    st.subheader("🚀 Train a New Model")

    model = NextWordMLP(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        hidden_dim=hidden_dim,
        block_size=context_len,
        num_layers=layers,
        activation=activation
    ).to(device)

    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None

    if st.button("Train Model"):
        st.info("🧠 Training started...")
        start = time.time()
        model = train_model(model, lines, stoi, epochs=epochs)
        st.session_state.trained_model = model
        st.success(f"✅ Training complete in {time.time() - start:.1f}s on {device}")

    user_input = st.text_input("Enter prompt text", "once upon a time")

    if st.button("Generate from Trained Model"):
        if st.session_state.trained_model is None:
            st.error("⚠️ Please train the model first.")
        else:
            with st.spinner("Generating..."):
                output = predict_next_words(
                    st.session_state.trained_model,
                    user_input,
                    k,
                    temperature,
                    stoi,
                    itos,
                    context_len
                )
                st.write("### 🧠 Generated Text:")
                st.success(output)

    st.markdown("---")
    st.info("""
    **Tips:**
    - Make sure the `stoi.json` & `itos.json` used for training are present in the checkpoints folder.
    - If you still see mismatches, copy the printed mismatch lines and paste here — I will point the exact dimension that differs.
    """)
