#  Context-Aware NLP Research Assistant

An advanced **Retrieval-Augmented Generation (RAG)** system for research paper analysis, featuring hybrid retrieval, semantic search, and AI-powered Q&A.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

## Features

- **Custom Fine-tuned Models**: Domain-adapted embedding model and LLM for scientific text
- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) search with learned re-ranking
- **Intelligent Q&A**: Context-aware question answering with source citations
- **Paper Summarization**: Multi-stage summarization and key insight extraction
- **Comprehensive Evaluation**: Quantified improvements over baseline models

## Architecture

```
User Query → Hybrid Retrieval (Dense + Sparse) → Re-ranking → Fine-tuned LLM → Answer + Citations
```

### Components

1. **Custom Embedding Model**: Fine-tuned sentence-transformer for scientific text
2. **Hybrid Retrieval**: FAISS (dense) + BM25 (sparse) + Cross-encoder (re-ranking)
3. **Fine-tuned LLM**: QLoRA-adapted Llama/Mistral for research Q&A
4. **Evaluation Framework**: Automated metrics and comparison benchmarks

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd NLP-Research-Assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Download Papers

Place your research papers (PDFs) in `data/papers/`

## Usage

### Running the Application

```bash
streamlit run src/ui/streamlit_app.py
```

### Training Custom Models

See notebooks for step-by-step training:
- `notebooks/02_embedding_finetuning.ipynb` - Custom embeddings
- `notebooks/03_llm_finetuning.ipynb` - Fine-tuned LLM

## Project Structure

```
NLP-Research-Assistant/
├── notebooks/           # Training and evaluation notebooks
├── src/
│   ├── data/           # Data loading and processing
│   ├── models/         # Model implementations
│   ├── training/       # Training scripts
│   ├── evaluation/     # Evaluation framework
│   └── ui/             # Streamlit interface
├── data/               # Papers and training data
├── models/             # Saved fine-tuned models
└── outputs/            # Reports and evaluations
```

## Technologies

- **LLMs**: Google Gemini, Llama 3.2, Mistral
- **Frameworks**: LangChain, HuggingFace Transformers
- **Vector DB**: FAISS
- **Training**: QLoRA, sentence-transformers
- **UI**: Streamlit

## License

MIT

## Author

Built as a learning project for GenAI and RAG systems.
