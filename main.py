import os
import re
import requests
import sqlite3
import threading
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
import mwparserfromhell
import logging
import chromadb
from collections import deque
from huggingface_hub import login

# LangServe and FastAPI imports
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field # Use pydantic_v1 for LangChain
from typing import List, Tuple, Dict, Any, Union
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import uvicorn

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_DIR = "one_piece_cache"
DB_PATH = os.path.join(CACHE_DIR, "one_piece_data.db")
CHROMA_DB_PATH = os.path.join(CACHE_DIR, "chroma_db")

# Model Selection
LLM_MODEL = "google/gemma-2-2b-it" # Ensure you have access and resources for this model
EMBED_MODEL = "intfloat/e5-small-v2"

# Hugging Face token - It's best to set this as an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN") # Removed default token for security
# If HF_TOKEN is None and your model is gated or private, login will fail.
# For public models, login might not be strictly necessary but good practice.

# Key One Piece categories to crawl
WIKI_CATEGORIES = {
    "Characters": ["Straw_Hat_Pirates", "Marines", "Yonko", "Seven_Warlords", "Worst_Generation"],
    "Devil_Fruits": ["Paramecia", "Zoan", "Logia"],
    "Locations": ["Islands", "Seas", "Grand_Line", "New_World"],
    "Story": ["Story_Arcs", "Sagas", "Events"],
    "Organizations": ["Pirates", "Crews", "Marines", "World_Government"],
    "Concepts": ["Haki", "Void_Century", "Ancient_Weapons", "Will_of_D"]
}

# Most important pages (prioritized)
CRUCIAL_PAGES = [
    "Monkey_D._Luffy", "Straw_Hat_Pirates", "One_Piece_(Manga)", "Eiichiro_Oda",
    "Devil_Fruit", "Haki", "Void_Century", "Gol_D._Roger", "Marines", "Yonko",
    "World_Government", "Grand_Line", "New_World", "One_Piece", "Will_of_D",
    "Poneglyphs", "Ancient_Weapons", "Roger_Pirates", "God_Valley_Incident",
    "Joy_Boy", "Sun_God_Nika", "Laugh_Tale", "Rocks_Pirates", "Revolutionary_Army",
    "Hito_Hito_no_Mi,_Model:_Nika", "Gomu_Gomu_no_Mi", "Five_Elders", "Im",
    "Marshall_D._Teach", "Blackbeard_Pirates", "Gura_Gura_no_Mi", "Yami_Yami_no_Mi"
]

# Processing Parameters
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP = 2
MAX_CONTEXT_CHUNKS = 10
SIMILARITY_THRESHOLD = 0.35
REFRESH_INTERVAL = 7 * 24 * 3600
CONVERSATION_HISTORY_LENGTH = 6 # This will be managed by RunnableWithMessageHistory's store

class OnePieceChatbot:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        if HF_TOKEN:
            try:
                login(token=HF_TOKEN)
                logging.info("Successfully logged into Hugging Face Hub.")
            except Exception as e:
                logging.error(f"Hugging Face Hub login failed: {e}. Public models may still work.")
        else:
            logging.warning("HF_TOKEN not set. Operations requiring authentication with Hugging Face Hub may fail.")

        self.db_conn = self._init_db()
        self.chroma_client, self.chroma_collection = self._init_chroma()

        self.data_lock = threading.Lock()
        self.processing_pages = set()
        self.initial_processing_done = threading.Event()

        logging.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        logging.info(f"Loading LLM tokenizer: {LLM_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        logging.info(f"Loading LLM model: {LLM_MODEL}")
        # Ensure GPU is available if you expect good performance
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        if device_map == "cpu":
            logging.warning("CUDA not available, loading LLM on CPU. This will be very slow.")
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32 # bfloat16 might not be supported on CPU
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.2
        )
        logging.info("Models loaded successfully.")
        
        # Start background data processing
        # For a production LangServe app, consider how this thread interacts with multiple workers if you scale.
        # For simplicity, we'll keep it as is.
        self.data_processing_thread = threading.Thread(target=self._process_wiki_data, daemon=True)
        self.data_processing_thread.start()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_data (
                title TEXT PRIMARY KEY,
                content TEXT,
                category TEXT,
                last_fetched REAL,
                page_links TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON wiki_data (category)")
        return conn

    def _init_chroma(self):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name="one_piece_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        return client, collection

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_wiki_page(self, title):
        url = f"https://onepiece.fandom.com/api.php?action=parse&page={title}&format=json&prop=wikitext|categories"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "parse" not in data: return None, [], None
        wikitext = data["parse"]["wikitext"]["*"]
        parsed = mwparserfromhell.parse(wikitext)
        for node in parsed.ifilter_templates():
            try: parsed.remove(node)
            except ValueError: pass
        links = [str(link.title).split("#")[0].strip() for link in parsed.ifilter_wikilinks() if ":" not in str(link.title) and len(str(link.title).split("#")[0].strip()) > 1]
        category = "Other"
        if "categories" in data["parse"]:
            categories_data = [cat["*"] for cat in data["parse"]["categories"]]
            for cat_type, cat_list in WIKI_CATEGORIES.items():
                if any(cat in categories_data for cat in cat_list):
                    category = cat_type
                    break
        text = parsed.strip_code().strip()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text, links, category

    def _fetch_category_pages(self, category):
        url = f"https://onepiece.fandom.com/api.php?action=query&list=categorymembers&cmtitle=Category:{category}&cmlimit=500&format=json" # Adjust cmlimit as needed
        response = requests.get(url, timeout=15)
        data = response.json()
        pages = []
        if "query" in data and "categorymembers" in data["query"]:
            for member in data["query"]["categorymembers"]:
                if "title" in member and ":" not in member["title"]: # Avoid "Category:", "File:", etc.
                    pages.append(member["title"])
        return pages

    def _process_wiki_data(self):
        logging.info("Starting initial data processing for crucial pages...")
        for page in CRUCIAL_PAGES:
            self._process_page(page)
        logging.info("Finished processing crucial pages.")

        logging.info("Starting data processing for categories...")
        for category_type, categories in WIKI_CATEGORIES.items():
            for category in categories:
                try:
                    pages = self._fetch_category_pages(category)
                    logging.info(f"Found {len(pages)} pages in category {category_type}:{category}")
                    for page in pages:
                        if page not in self.processing_pages and page not in CRUCIAL_PAGES: # Avoid reprocessing
                            self._process_page(page)
                except Exception as e:
                    logging.error(f"Error processing category {category}: {e}")
        logging.info("Initial data processing from categories complete.")
        self.initial_processing_done.set()
        logging.info(f"Background process: Loaded {self.chroma_collection.count()} chunks of One Piece knowledge.")

        while True:
            time.sleep(REFRESH_INTERVAL)
            logging.info("Starting refresh cycle...")
            cur = self.db_conn.execute("SELECT title FROM wiki_data ORDER BY last_fetched ASC LIMIT 100") # Refresh oldest 100
            pages_to_refresh = [row[0] for row in cur.fetchall()]
            for page in pages_to_refresh:
                self._process_page(page, force_refresh=True)
            logging.info("Refresh cycle complete.")

    def _process_page(self, title, force_refresh=False):
        with self.data_lock:
            if title in self.processing_pages: return
            self.processing_pages.add(title)

        try:
            if not force_refresh:
                cur = self.db_conn.execute("SELECT last_fetched FROM wiki_data WHERE title = ?", (title,))
                result = cur.fetchone()
                if result and (time.time() - result[0] < REFRESH_INTERVAL):
                    # logging.info(f"Skipping {title}, recently fetched.")
                    self.processing_pages.discard(title) # Use discard for sets
                    return

            logging.info(f"Fetching and processing page: {title}")
            content, links, category = self._fetch_wiki_page(title)
            if not content:
                self.processing_pages.discard(title)
                return

            self.db_conn.execute(
                "INSERT OR REPLACE INTO wiki_data VALUES (?, ?, ?, ?, ?)",
                (title, content, category, time.time(), ','.join(links))
            )
            self.db_conn.commit()

            chunks = self._chunk_text(content, title)
            if chunks:
                embeddings = self.embedder.encode(chunks, convert_to_tensor=False).tolist()
                ids = [f"{title}::{i}" for i in range(len(chunks))]
                metadatas = [{"source": title, "category": category} for _ in chunks]
                self.chroma_collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
                logging.info(f"Processed and upserted {title}: {len(chunks)} chunks")
            

        except Exception as e:
            logging.error(f"Error processing {title}: {e}")
        finally:
            self.processing_pages.discard(title)

    def _chunk_text(self, text, title):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_sentences = [], []
        token_count = 0
        for sentence in sentences:
            tokens = len(self.tokenizer.tokenize(sentence)) # Use LLM tokenizer for more accurate count
            if token_count + tokens > CHUNK_SIZE_TOKENS and current_sentences:
                chunks.append(" ".join(current_sentences))
                if CHUNK_OVERLAP > 0 and len(current_sentences) > CHUNK_OVERLAP:
                    current_sentences = current_sentences[-CHUNK_OVERLAP:]
                    token_count = sum(len(self.tokenizer.tokenize(s)) for s in current_sentences)
                else:
                    current_sentences, token_count = [], 0
            current_sentences.append(sentence)
            token_count += tokens
        if current_sentences: chunks.append(" ".join(current_sentences))
        return [chunk for chunk in chunks if len(self.tokenizer.tokenize(chunk)) > 20] # Min chunk token length

    def _format_chat_history_for_prompt(self, chat_history: List[Tuple[str, str]]):
        if not chat_history: return ""
        history_str = "Recent conversation:\n"
        for q, a in chat_history:
            history_str += f"User: {q}\nAssistant: {a}\n"
        return history_str
    
    def _interpret_query_with_history(self, query: str, formatted_history: str):
        if len(query.split()) < 3 or query.lower().startswith(("and ", "what about ")):
            if not formatted_history: return query
            prompt = f"""
Based on this conversation history:
{formatted_history}

The user asked a follow-up question: "{query}"
Please interpret this as a complete, standalone question about One Piece.
Only provide the complete question and nothing else:"""
            try:
                response = self.generator(prompt)[0]["generated_text"].split("Only provide the complete question and nothing else:")[-1].strip()
                logging.info(f"Interpreted '{query}' as '{response}' using history.")
                return response
            except Exception as e:
                logging.error(f"Error interpreting query: {e}")
                return query # Fallback to original query
        return query

    def _find_relevant_chunks(self, query: str, formatted_history: str):
        interpreted_query = self._interpret_query_with_history(query, formatted_history)
        
        # Enhanced query expansion
        if any(topic in interpreted_query.lower() for topic in ["joy boy", "nika", "luffy", "d clan", "devil fruit"]):
            if "joy boy" in interpreted_query.lower() and "nika" in interpreted_query.lower():
                interpreted_query += " Hito Hito no Mi Model Nika Devil Fruit connection"
            if "blackbeard" in interpreted_query.lower() and "devil fruit" in interpreted_query.lower():
                interpreted_query += " multiple devil fruits Yami Yami no Mi Gura Gura no Mi"
            if "gorosei" in interpreted_query.lower() or "im" in interpreted_query.lower():
                interpreted_query += " Five Elders Empty Throne World Government"

        query_embedding = self.embedder.encode(interpreted_query).tolist()
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=MAX_CONTEXT_CHUNKS,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks, sources = [], []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                if similarity >= SIMILARITY_THRESHOLD:
                    chunks.append(doc)
                    sources.append(results["metadatas"][0][i]["source"])
        return chunks, list(set(sources)), interpreted_query

    def get_answer(self, question: str, chat_history_tuples: List[Tuple[str, str]]):
        """
        Generates an answer based on the question and conversation history.
        This is the core logic that will be part of the LangChain Runnable.
        """
        if not self.initial_processing_done.is_set():
            logging.warning("Initial data processing is not yet complete. RAG results might be suboptimal.")
        
        formatted_history = self._format_chat_history_for_prompt(chat_history_tuples)
        chunks, sources, interpreted_query = self._find_relevant_chunks(question, formatted_history)

        prompt = f"""You are an expert on the One Piece manga and anime. Answer the following question based on the provided context and conversation history.

{formatted_history}

Context information from One Piece Wiki:
{" ".join(chunks) if chunks else "No specific context found for this query. Answer based on general One Piece knowledge."}

Interpreted Question: {interpreted_query}
Original Question: {question}

Provide a detailed, accurate answer. If the context doesn't contain enough information, state that the context is limited but still try to provide a helpful answer based on general One Piece lore if possible. Include specific details and explain connections clearly.
IMPORTANT: Your answer must be directly useful. Do not say "Based on the context..." or "To answer your question...". Start immediately with the answer.
"""
        try:
            response_full = self.generator(prompt)[0]["generated_text"]
            # More robust answer extraction
            answer_marker = "Start immediately with the answer." # Or any other reliable marker after your instructions
            if answer_marker in response_full:
                answer = response_full.split(answer_marker, 1)[-1].strip()
            else: # Fallback if marker is not found (e.g., model doesn't follow instructions perfectly)
                # Try to remove the prompt part
                answer = response_full.replace(prompt, "").strip() # This is a bit crude
                # A more refined approach might be needed if the model includes preamble

            # Further cleaning
            answer = re.sub(r"^(Answer:|Okay, here's the answer based on the information:|Certainly, based on the provided context and your question about One Piece:)\s*", "", answer, flags=re.IGNORECASE).strip()


        except Exception as e:
            logging.error(f"Error during LLM generation: {e}")
            answer = "I encountered an issue trying to generate a response. Please try again."

        if sources:
            sources_str = ", ".join(list(set(sources))[:5])
            return f"{answer}\n\nSources: {sources_str}"
        return answer

# --- LangServe Setup ---

# 1. Initialize the chatbot (this will load models and start data processing)
# This instance will be shared across requests in a single-worker setup.
# For multi-worker, each worker would have its own instance.
chatbot_instance = OnePieceChatbot()

# Wait a bit for some initial data to be processed, or for crucial pages at least.
# For a real deployment, you might have a readiness probe.
logging.info("Waiting for initial crucial data processing to complete (max 60s)...")
if not chatbot_instance.initial_processing_done.wait(timeout=180): # Increased timeout
    logging.warning("Initial data processing might still be ongoing in the background. Chatbot is starting anyway.")
else:
    logging.info(f"Initial data processing complete. ChromaDB has {chatbot_instance.chroma_collection.count()} chunks.")


# 2. Pydantic models for input and output
# For LangServe's input, the chat_history should be a list of BaseMessage
class ChatInput(BaseModel):
    question: str
    # When using RunnableWithMessageHistory, the 'chat_history' key receives BaseMessage objects
    chat_history: List[BaseMessage] = Field(default_factory=list, extra={"widget_type": "chat"})


class ChatOutput(BaseModel):
    answer: str
    # We could add sources or other metadata here if needed

# 3. Store for conversation histories (in-memory for this example)
# For production, consider using RedisChatMessageHistory or another persistent store.
message_history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_history_store:
        message_history_store[session_id] = InMemoryChatMessageHistory() # Langchain's InMemoryChatMessageHistory
    return message_history_store[session_id]

# 4. Create the core LangChain Runnable

def _extract_chat_history_tuples(lc_chat_history: List[BaseMessage]) -> List[Tuple[str, str]]:
    """
    Converts a list of LangChain BaseMessage objects into a list of (human_message, ai_message) tuples.
    Assumes messages alternate HumanMessage, AIMessage.
    """
    formatted_history_tuples = []
    user_msg_content = None
    for msg in lc_chat_history:
        if isinstance(msg, HumanMessage):
            user_msg_content = msg.content
        elif isinstance(msg, AIMessage) and user_msg_content is not None:
            formatted_history_tuples.append((user_msg_content, msg.content))
            user_msg_content = None # Reset for the next pair
    return formatted_history_tuples

# The chain now needs to correctly prepare inputs for chatbot_instance.get_answer
# The input to this chain will be a dictionary with 'question' and 'chat_history' (List[BaseMessage])
# We need to map this to the 'question' (str) and 'chat_history_tuples' (List[Tuple[str,str]])
# expected by chatbot_instance.get_answer.

# This RunnablePassthrough.assign takes the raw input from LangServe (which includes chat_history as BaseMessages)
# and transforms it into the format expected by get_answer.
# The 'question' is passed through, and 'chat_history_tuples' is created from 'chat_history'.
core_logic_chain = RunnablePassthrough.assign(
    chat_history_tuples=RunnableLambda(_extract_chat_history_tuples).with_types(
        input_type=List[BaseMessage], # Input to this lambda is the chat_history from the main input
        output_type=List[Tuple[str,str]]
    )
).assign(
    # The 'answer' key holds the final result from the chatbot's get_answer method
    answer=RunnableLambda(
        lambda x: chatbot_instance.get_answer(x["question"], x["chat_history_tuples"])
    ).with_types(input_type=Dict[str, Any], output_type=str) # The input to this lambda is a dict with 'question' and 'chat_history_tuples'
).with_types(
    # Explicitly set the output type of this core_logic_chain to match ChatOutput
    # This runnable will return a dictionary like {"question": ..., "chat_history_tuples": ..., "answer": ...}
    # We want it to directly output just the 'answer' string for the final chain.
    output_type=str # The last assign directly produces the 'answer' string for the output_messages_key below
)

# Wrap with history management
# The input to this chain will be a dictionary with "question" (str) and "chat_history" (List[BaseMessage])
# The output will be the 'answer' string.
conversational_rag_chain = RunnableWithMessageHistory(
    core_logic_chain, # The core runnable that takes raw input and produces the answer string
    get_session_history,    # Function to get/create session history
    input_messages_key="question", # Key in the input dict for the user's question
    history_messages_key="chat_history", # Key in the input dict where history (List[BaseMessage]) will be injected
    output_messages_key="answer" # The final answer string produced by core_logic_chain is mapped to this key
).with_types(
    input_type=ChatInput,  # The API endpoint will expect ChatInput
    output_type=str        # The API endpoint will return a string (the answer)
)


# 5. FastAPI App
app = FastAPI(
    title="One Piece RAG Chatbot",
    version="1.0",
    description="A RAG chatbot for One Piece encyclopedia using LangServe",
)

# Add the route for your conversational chain
# This will expose endpoints like /onepiece_chat/invoke, /onepiece_chat/playground, etc.
add_routes(
    app,
    conversational_rag_chain,
    path="/onepiece_chat",
    # Input/Output types are inferred or explicitly set by .with_types() on the chain
    # input_type=ChatInput,  # Not needed here, as it's already on the chain
    # output_type=ChatOutput # Not needed here, as the chain's output is directly a string which is handled by output_messages_key
    config_keys=["session_id"] # Expose session_id for configurable history
)

# Optional: Add a root path or other utility endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to the One Piece RAG Chatbot API!",
        "docs": "/docs",
        "playground": "/onepiece_chat/playground/"
    }

if __name__ == "__main__":
    # Ensure HF_TOKEN is set if needed by your models
    if not HF_TOKEN and (LLM_MODEL.startswith("google/") or LLM_MODEL.startswith("meta-llama/")): # Gemma and Llama models often require auth
        logging.warning(
            f"HF_TOKEN is not set. Model {LLM_MODEL} might require authentication. "
            "Set the HF_TOKEN environment variable with your Hugging Face access token."
        )

    logging.info("Starting Uvicorn server...")
    # Make sure the host and port are configurable for deployment
    # For local testing:
    uvicorn.run(app, host="0.0.0.0", port=8000)