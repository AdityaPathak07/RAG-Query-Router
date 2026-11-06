# RAG Query Router

A Retrieval-Augmented Generation (RAG) app that **routes** a user's question to the
right knowledge base, retrieves relevant content, and answers with **citations and source links**.

## How it works

```
User question
     │
     ▼
[1] Router (LLM) ──► picks one collection
     │
     ▼
[2] Qdrant similarity search in that collection (top 4 chunks)
     │
     ▼
[3] LLM answers using only the retrieved context, citing [1], [2]...
     │
     ▼
[4] References printed from chunk metadata (title + URL)
```

1. **Router:** An LLM with structured output (`Literal[...]`) chooses exactly one collection.
2. **Retrieval:** Only that collection is searched, which keeps results focused.
3. **Answer:** The LLM answers only from the retrieved context. If the answer isn't there, it says it doesn't know.
4. **References:** Source URLs come from document metadata, not from the LLM, so they are always accurate.

## Collections

| Collection | Source |
|---|---|
| `generative_ai` | What is Generative AI (GeeksforGeeks) |
| `ai_in_robotics` | AI in Robotics (GeeksforGeeks) |
| `machine_learning` | Introduction to Machine Learning (GeeksforGeeks) |

## Tech stack

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://platform.openai.com/) (`gpt-4o-mini`, `text-embedding-3-small`)
- [Qdrant](https://qdrant.tech/) vector database

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/rag-query-router.git
cd rag-query-router
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Add your API key

Copy `.env.example` to `.env` and fill in your key:

```
OPENAI_API_KEY=your-key-here
```

### 4. Index the data (first run only)

In `app.py`, set:

```python
INDEX_DOCUMENTS = True
```

Run once, then set it back to `False`:

```bash
python app.py
```

### 5. Ask questions

```bash
python app.py
```

## Example

```
Enter your query: How do robots use AI to see objects?
Routed to: ai_in_robotics

Answer:
 Robots use computer vision and machine learning to detect and identify objects [1][2].

References:
- Artificial Intelligence in Robotics - GeeksforGeeks
  https://www.geeksforgeeks.org/artificial-intelligence/artificial-intelligence-in-robotics/
```

## Customize

- **Add a new topic:** add a URL to `SOURCES`, add its name to the `Literal[...]` in `Route`, and describe it in the router prompt.
- **Change retrieval size:** edit `k=4` in `similarity_search`.

## Limitations

- The router must pick one of the listed collections, even for unrelated questions.
- Answers are limited to the content of the indexed pages.

## Security

Never commit your `.env` file or API key. It is listed in `.gitignore`.
