import os
import re
import requests
import sqlite3
import threading
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
import mwparserfromhell
import logging
import chromadb
from collections import deque
from huggingface_hub import login

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any

# LangChain imports (for internal use only, not for API models)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

import uvicorn

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_DIR = "one_piece_cache"
DB_PATH = os.path.join(CACHE_DIR, "one_piece_data.db")
CHROMA_DB_PATH = os.path.join(CACHE_DIR, "chroma_db")

# Model Selection
LLM_MODEL = "Qwen/Qwen3-8B"
EMBED_MODEL = "intfloat/e5-small-v2"

HF_TOKEN = os.environ.get("HF_TOKEN")

WIKI_CATEGORIES = {
    "Characters": ["Straw_Hat_Pirates", "Marines", "Yonko", "Seven_Warlords", "Worst_Generation"],
    "Devil_Fruits": ["Paramecia", "Zoan", "Logia"],
    "Locations": ["Islands", "Seas", "Grand_Line", "New_World"],
    "Story": ["Story_Arcs", "Sagas", "Events"],
    "Organizations": ["Pirates", "Crews", "Marines", "World_Government"],
    "Concepts": ["Haki", "Void_Century", "Ancient_Weapons", "Will_of_D"]
}

CRUCIAL_PAGES = [
    "Monkey_D._Luffy", "Straw_Hat_Pirates", "One_Piece_(Manga)", "Eiichiro_Oda",
    "Devil_Fruit", "Haki", "Void_Century", "Gol_D._Roger", "Marines", "Yonko",
    "World_Government", "Grand_Line", "New_World", "One_Piece", "Will_of_D",
    "Poneglyphs", "Ancient_Weapons", "Roger_Pirates", "God_Valley_Incident",
    "Joy_Boy", "Sun_God_Nika", "Laugh_Tale", "Rocks_Pirates", "Revolutionary_Army",
    "Hito_Hito_no_Mi,_Model:_Nika", "Gomu_Gomu_no_Mi", "Five_Elders", "Im",
    "Marshall_D._Teach", "Blackbeard_Pirates", "Gura_Gura_no_Mi", "Yami_Yami_no_Mi"
]

CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP = 2
MAX_CONTEXT_CHUNKS = 10
SIMILARITY_THRESHOLD = 0.35
REFRESH_INTERVAL = 7 * 24 * 3600
CONVERSATION_HISTORY_LENGTH = 6

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

        logging.info(f"Loading LLM model (4-bit quantized): {LLM_MODEL}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
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
        logging.info("Models loaded successfully (quantized).")
        
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
        url = f"https://onepiece.fandom.com/api.php?action=query&list=categorymembers&cmtitle=Category:{category}&cmlimit=500&format=json"
        response = requests.get(url, timeout=15)
        data = response.json()
        pages = []
        if "query" in data and "categorymembers" in data["query"]:
            for member in data["query"]["categorymembers"]:
                if "title" in member and ":" not in member["title"]:
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
                        if page not in self.processing_pages and page not in CRUCIAL_PAGES:
                            self._process_page(page)
                except Exception as e:
                    logging.error(f"Error processing category {category}: {e}")
        logging.info("Initial data processing from categories complete.")
        self.initial_processing_done.set()
        logging.info(f"Background process: Loaded {self.chroma_collection.count()} chunks of One Piece knowledge.")

        while True:
            time.sleep(REFRESH_INTERVAL)
            logging.info("Starting refresh cycle...")
            cur = self.db_conn.execute("SELECT title FROM wiki_data ORDER BY last_fetched ASC LIMIT 100")
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
                    self.processing_pages.discard(title)
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
            tokens = len(self.tokenizer.tokenize(sentence))
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
        return [chunk for chunk in chunks if len(self.tokenizer.tokenize(chunk)) > 20]

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
                logging.info(f"Interpreted '{query}' as '{response}' as '{response}' using history.")
                return response
            except Exception as e:
                logging.error(f"Error interpreting query: {e}")
                return query
        return query

    def _find_relevant_chunks(self, query: str, formatted_history: str):
        interpreted_query = self._interpret_query_with_history(query, formatted_history)
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
            answer_marker = "Start immediately with the answer."
            if answer_marker in response_full:
                answer = response_full.split(answer_marker, 1)[-1].strip()
            else:
                answer = response_full.replace(prompt, "").strip()
            answer = re.sub(r"^(Answer:|Okay, here's the answer based on the information:|Certainly, based on the provided context and your question about One Piece:)\s*", "", answer, flags=re.IGNORECASE).strip()
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}")
            answer = "I encountered an issue trying to generate a response. Please try again."

        if sources:
            sources_str = ", ".join(list(set(sources))[:5])
            return answer, list(set(sources)), sources_str
        return answer, [], ""

# --- FastAPI Setup ---

# Initialize the chatbot (shared instance)
chatbot_instance = OnePieceChatbot()

logging.info("Waiting for initial crucial data processing to complete (max 60s)...")
if not chatbot_instance.initial_processing_done.wait(timeout=180):
    logging.warning("Initial data processing might still be ongoing in the background. Chatbot is starting anyway.")
else:
    logging.info(f"Initial data processing complete. ChromaDB has {chatbot_instance.chroma_collection.count()} chunks.")

# Pydantic models for input and output
class ChatInput(BaseModel):
    question: str
    history: List[Tuple[str, str]] = []

class ChatOutput(BaseModel):
    answer: str
    sources: List[str] = []

app = FastAPI(
    title="One Piece RAG Chatbot",
    version="1.0",
    description="A RAG chatbot for One Piece encyclopedia using LangChain",
)

@app.post("/chat", response_model=ChatOutput)
def chat(input: ChatInput):
    answer, sources, _ = chatbot_instance.get_answer(input.question, input.history)
    return ChatOutput(answer=answer, sources=sources)

@app.get("/")
def root():
    return {
        "message": "Welcome to the One Piece RAG Chatbot API!",
        "docs": "/docs"
    }

if __name__ == "__main__":
    if not HF_TOKEN and (LLM_MODEL.startswith("google/") or LLM_MODEL.startswith("meta-llama/")):
        logging.warning(
            f"HF_TOKEN is not set. Model {LLM_MODEL} might require authentication. "
            "Set the HF_TOKEN environment variable with your Hugging Face access token."
        )
    logging.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
