import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from typing import Literal
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_URL = "http://localhost:6333/"
INDEX_DOCUMENTS = True  # set True only the first time to store the URLs

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

SOURCES = {
    "generative_ai": "https://www.geeksforgeeks.org/artificial-intelligence/what-is-generative-ai/",
    "ai_in_robotics": "https://www.geeksforgeeks.org/artificial-intelligence/artificial-intelligence-in-robotics/",
    "machine_learning": "https://www.geeksforgeeks.org/machine-learning/introduction-machine-learning/",
}

# ---------- One-time indexing ----------
if INDEX_DOCUMENTS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for name, url in SOURCES.items():
        data = WebBaseLoader(url).load()
        splits = splitter.split_documents(data)
        QdrantVectorStore.from_documents(
            splits,
            embedding=embeddings,
            collection_name=name,
            url=QDRANT_URL,
        )
        print(f"Indexed: {name}")


class Route(BaseModel):
    collection: Literal["generative_ai", "ai_in_robotics", "machine_learning"]

router_llm = llm.with_structured_output(Route)

user_query = input("Enter your query: ")

route_prompt = f"""Choose the best collection for this question.

Collections:
- generative_ai: what generative AI is, how it works
- ai_in_robotics: AI used in robots and robotics
- machine_learning: introduction to machine learning

Question: {user_query}
"""
collection_name = router_llm.invoke(route_prompt).collection
print("Routed to:", collection_name)

# ---------- Step 2: Retrieve from that collection ----------
retrieval = QdrantVectorStore.from_existing_collection(
    collection_name=collection_name,
    embedding=embeddings,
    url=QDRANT_URL,
)

chunks = retrieval.similarity_search(query=user_query, k=4)
context = "\n\n".join(c.page_content for c in chunks)

# ---------- Step 3: Answer using the retrieved context ----------
answer_prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {user_query}
"""

result = llm.invoke(answer_prompt)
print("\nAnswer:\n", result.content)

# ---------- Step 4: Print references ----------
print("\nReferences:")
seen = set()
for i, c in enumerate(chunks, start=1):
    url = c.metadata.get("source", SOURCES[collection_name])
    title = c.metadata.get("title", "Untitled").strip()
    if url not in seen:          # avoid printing the same link 4 times
        seen.add(url)
        print(f"- {title}\n  {url}")