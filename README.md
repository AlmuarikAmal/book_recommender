# Semantic Book Recommender with LLMs 📚

This repository contains all the code and data used to follow the [freeCodeCamp course](https://www.youtube.com/watch?v=Q7mS1VHm3Yw) titled **“Build a Semantic Book Recommender with LLMs – Full Course.”**

In this project, we build a smart book recommendation system that uses **Large Language Models (LLMs)** and **semantic search** to find meaningful book matches based on user queries — like _"a book about a person seeking revenge"_ — rather than relying solely on keyword matching.

---

## Dataset Used

- Dataset from Kaggle: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata  
- Files like `book_cleaned.csv`, `books_with_emotions.csv`, and `books_with_categories.csv` are generated as part of the pipeline.

---

## Dependencies

This project uses:

- `pandas`, `matplotlib`, `seaborn`
- `gradio`
- `langchain`, `transformers`, `kagglehub`
- `python-dotenv`
- `notebook`, `ipywidgets`
- `chromadb`, `langchain-community`

> All dependencies are listed in the `requirements.txt` file.

---

## What I Learned

This project consists of five key components:

### 1. Data Cleaning
- Notebook: `data-exploration.ipynb`
- Explore and preprocess book data (e.g., clean descriptions, remove nulls)

### 2. Semantic Search with Vector Embeddings
- Notebook: `vector-search.ipynb`
- Build a vector database that lets users search for books using natural language queries

### 3. Zero-shot Text Classification
- Notebook: `text-classification.ipynb`
- Classify books as *fiction* or *non-fiction* using LLM zero-shot classification

### 4. Sentiment & Emotion Analysis
- Notebook: `sentiment-analysis.ipynb`
- Extract emotional tone (e.g., joyful, suspenseful, sad) from book descriptions

### 5. Gradio Web App
- Script: `gradio-dashboard.py`
- Build an interactive web app to recommend books using everything above

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AlmuarikAmal/book_recommender.git
cd book_recommender
```

### 2. Set up your environment
```bash 
python -m venv .venv
source .venv/bin/activate    # for mac 
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
Create a .env file in the root directory with 
OPENAI_API_KEY=your_open_AI_key_here


## Running the App

Once your environment is ready and `.env` is configured with your OpenAI API key, you can launch the interactive book recommender app using Gradio:

```bash
python gradio-dashboard.py
```
