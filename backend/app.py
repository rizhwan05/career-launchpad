import os
import logging
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import torch

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Semantic relevance model
semantic_model = SentenceTransformer(EMBEDDING_MODEL)

# Reference queries to compare against
reference_queries = [
    "What is the average package?",
    "Eligibility for placements",
    "Interview process at TCS",
    "Resume tips for freshers",
    "What is CTC in college placements?",
    "Companies visiting our campus",
    "How to crack campus placement?",
    "Which companies offer internships?",
    "Campus recruitment drive",
    "Skills required for Google",
    "Job role in Infosys"
]

# Precompute reference embeddings
reference_embeddings = semantic_model.encode(reference_queries, convert_to_tensor=True)


# -----------------------------------
# Configuration
# -----------------------------------
os.environ["USE_TF"] = "0"  # Disable TensorFlow usage for Sentence Transformers

DATA_PATH = "placement_data.csv"
LLAMA_MODEL_PATH = "model/llama-3.2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K_RESULTS = 3

# -----------------------------------
# Initialize FastAPI App
# -----------------------------------
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

# -----------------------------------
# Load Dataset and Create Vector Store
# -----------------------------------
df = pd.read_csv(DATA_PATH)

documents = [
    Document(page_content=row["Answer"], metadata={"question": row["Question"]})
    for idx, row in df.iterrows()
]

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.from_documents(documents, embedding_model)

# -----------------------------------
# Load Llama Model
# -----------------------------------
if os.path.exists(LLAMA_MODEL_PATH):
    llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    temperature=0.7,
    max_tokens=1024,
    n_ctx=1024,                 # Lower if your input size is small
    n_batch=256,                # Increase this if RAM allows (try 256, 512)
    n_threads=os.cpu_count(),  # Fully utilize your CPU
    verbose=False
    )
    logging.info("Llama model loaded successfully.")
else:
    llm = None
    logging.warning("Llama model not found. Fallback will not work.")

# -----------------------------------
# Prompt Templates
# -----------------------------------
query_prompt_template = PromptTemplate.from_template("""
You are a professional Placement Assistant for college students.

When answering:
- Be clear and concise.
- Stick to relevant, general placement information.
- If the query is broad (e.g., "Adobe salary"), suggest more specific sub-questions.
- Avoid repeating disclaimers.

Here is the user query:
{input}

Respond:
""")

summary_prompt_template = """
You are a professional Placement Assistant for a college.
ONLY based on the provided information below, answer the user query.
DO NOT make assumptions or add new examples.
If information is insufficient, politely say so.

Information:
{context}

User Query:
{query}

Answer:"""

# -----------------------------------
# Request Schema
# -----------------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------------
# Helper Functions
# -----------------------------------
THRESHOLD_SCORE = 60.0  # Accept only results better than this similarity score (you can tweak this)

def search_db(query, top_k=TOP_K_RESULTS):
    """Search the FAISS DB and return matched documents above a similarity threshold."""
    results = db.similarity_search_with_score(query, k=top_k)
    valid = []

    logging.info(f"Raw search results for query '{query}': {[(doc.page_content, score) for doc, score in results]}")

    for doc, score in results:
        if score < THRESHOLD_SCORE:  # ✅ Accept only if the score is strong enough
            valid.append(doc.page_content.strip())

    logging.info(f"Filtered valid results for query '{query}': {valid}")

    return valid if valid else None




import re

def fallback_llm(query):
    if not llm:
        return "LLM model not available."

    prompt = query_prompt_template.format(input=query)
    response = llm.invoke(prompt).strip()

    # Remove known disclaimer phrases (if they exist in model output)
    response = re.sub(r"\(?Note.*verified.*placement.*records\)?\.?", "", response, flags=re.IGNORECASE).strip()

    return "**(Based on general knowledge — not officially verified)**\n\n" + response



def summarize_results(results, user_query):
    """Combine multiple database results meaningfully using Llama."""
    if not results:
        return None  # No results, signal to fallback

    combined_context = "\n\n".join(results)
    
    summarization_prompt = f"""
You are a professional Placement Assistant. ONLY based on the following information, answer the user's query. DO NOT add external examples.

Information:
{combined_context}

User Query:
{user_query}

Answer:"""

    if llm:
        response = llm.invoke(summarization_prompt)
        return response.strip()
    else:
        # If LLM not available, just return combined context
        return combined_context

def is_relevant_query(query: str) -> bool:
    query = query.strip()
    if not query:
        return False

    # First: keyword pattern matching
    keywords = [
        "placement", "package", "salary", "internship", "interview", "resume",
        "recruiter", "eligibility", "campus", "drive", "career", "company",
        "hiring", "selection", "offer", "ctc", "college", "exam", "job",
        "skills", "qualifications", "role", "recruitment", "profile", "backlog"
    ]
    keyword_pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
    keyword_found = re.search(keyword_pattern, query.lower()) is not None

    # Second: semantic similarity
    try:
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, reference_embeddings)
        max_score = torch.max(cosine_scores).item()
        logging.info(f"Semantic similarity score: {max_score:.2f}")

        # Accept if score > 0.3 or keywords are found
        if max_score > 0.3 or keyword_found:
            return True
        else:
            return False
    except Exception as e:
        logging.warning(f"Semantic similarity check failed: {e}")
        # Fallback: only use keyword presence
        return keyword_found


# -----------------------------------
# API Endpoint
# -----------------------------------
@app.post("/query/")
async def handle_query(request: QueryRequest):
    user_query = request.query.strip()
    logging.info(f"Received user query: {user_query}")

    # Step 0: Early filter — Block only truly irrelevant queries
    if not is_relevant_query(user_query):
        logging.info("Query rejected: Not placement or career-related.")
        return {
            "source": "filter",
            "answers": [
                "**This assistant is focused only on placement, company, college, and career-related queries.**"
            ]
        }

    # Step 1: Search in database
    results = search_db(user_query)

    # Step 2: If valid results found, summarize them
    if results:
        logging.info("Database results found. Summarizing response.")
        final_response = summarize_results(results, user_query)
        final_response = "**(Based on verified placement data)**\n\n" + final_response
        return {"source": "db+llm", "answers": [final_response]}

    # Step 3: Use fallback LLM for general relevant queries
    logging.info("No database results found. Falling back to Llama.")
    fallback_response = fallback_llm(user_query)
    return {"source": "llm", "answers": [fallback_response]}



