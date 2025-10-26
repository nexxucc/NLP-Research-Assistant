# 🚀 Quick Start Guide

Get your NLP Research Assistant running in 5 minutes!

---

## ⚡ Setup (3 minutes)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Get API Key

1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key

### 3. Configure

Create a `.env` file:

```bash
GOOGLE_API_KEY=your_key_here
```

---

## 📄 Add Papers (1 minute)

```bash
# Create papers directory
mkdir -p data/papers

# Copy your PDF research papers into data/papers/
# Example: copy your drone paper
cp ~/Downloads/drones_research.pdf data/papers/
```

---

## 🔨 Build Index (1 minute)

```bash
python scripts/build_index.py
```

You'll see:
```
Loading PDF documents... ✓
Creating chunks... ✓
Generating embeddings... ✓
Building FAISS index... ✓
Index saved!
```

---

## 🌐 Launch App (<1 minute)

```bash
streamlit run src/ui/streamlit_app.py
```

Browser opens automatically at `http://localhost:8501`

---

## 💬 Start Asking Questions!

1. **Enter your API key** in the sidebar
2. **Click "Load Index"**
3. **Ask questions** like:
   - "What is the main contribution?"
   - "What sensors were used?"
   - "What were the results?"

Get answers with source citations!

---

## 🎯 Example Session

```python
# Or use Python directly
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load_index("data/faiss_index")

result = pipeline.query("What is 3D mapping?")
print(result['answer'])
```

---

## ❓ Troubleshooting

**"No PDF files found"**
→ Add PDFs to `data/papers/`

**"API key error"**
→ Check your `.env` file has correct key

**"Module not found"**
→ Run `pip install -r requirements.txt`

**"Index not found"**
→ Run `python scripts/build_index.py` first

---

## 🎓 Next Steps

- **Learn how it works**: Check out `notebooks/01_understanding_embeddings.ipynb`
- **Try hybrid retrieval**: See `src/models/hybrid_retriever.py`
- **Evaluate performance**: Use `src/evaluation/metrics.py`
- **Read full docs**: See `README.md`

---

**You're all set! Start exploring your research papers with AI! 🎉**
