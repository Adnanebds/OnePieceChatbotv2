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

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool # Import for running sync code in async app

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_DIR = "one_piece_cache"
DB_PATH = os.path.join(CACHE_DIR, "one_piece_data.db")
CHROMA_DB_PATH = os.path.join(CACHE_DIR, "chroma_db")

# Model Selection
LLM_MODEL = "google/gemma-2-2b-it"
EMBED_MODEL = "intfloat/e5-small-v2"

# Hugging Face token
# It's recommended to pass this via environment variables in production
HF_TOKEN = os.environ.get("HF_TOKEN")

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
MAX_CONTEXT_CHUNKS = 10 # Increased from 8 to include more context
SIMILARITY_THRESHOLD = 0.35
REFRESH_INTERVAL = 7 * 24 * 3600 # Weekly refresh
CONVERSATION_HISTORY_LENGTH = 6 # Enhanced for better context

class OnePieceChatbot:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Login to Hugging Face
        if HF_TOKEN:
            try:
                login(token=HF_TOKEN)
                logging.info("Successfully logged into Hugging Face.")
            except Exception as e:
                 logging.warning(f"Hugging Face login failed: {e}. Proceeding without explicit login.")


        # Initialize database and vector store
        self.db_conn = self._init_db()
        self.chroma_client, self.chroma_collection = self._init_chroma()

        # Initialize threading control
        self.data_lock = threading.Lock()
        self.processing_pages = set()
        self.initial_processing_done = threading.Event() # Event to signal initial data load

        # Initialize models - These are memory intensive, loaded once
        try:
            self.embedder = SentenceTransformer(EMBED_MODEL)
            logging.info(f"Loaded SentenceTransformer model: {EMBED_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            logging.info(f"Loaded Tokenizer: {LLM_MODEL}")
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto", # Automatically select device (GPU if available)
                torch_dtype=torch.bfloat16 # Use bfloat16 for efficiency
            )
            logging.info(f"Loaded LLM Model: {LLM_MODEL}")

            # Initialize text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=500, # Increased from 300 to avoid cut-off answers
                temperature=0.2,
                do_sample=True, # Set to True to match the temperature setting
                repetition_penalty=1.2 # Added to reduce repetition
            )
            logging.info("Initialized text generation pipeline.")

        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            # Depending on requirements, you might raise the exception or handle it gracefully
            # For deployment, you might want to ensure models load or the app fails early
            raise SystemExit("Failed to load models, exiting.") from e

        # Conversation memory (per instance, this will be global for the API)
        # A more complex app might manage history per user session
        self.conversation_history = deque(maxlen=CONVERSATION_HISTORY_LENGTH)

        # Start background data processing thread
        logging.info("Starting background data processing thread...")
        thread = threading.Thread(target=self._process_wiki_data, daemon=True)
        thread.start()
        logging.info("Background data processing thread started.")


    def _init_db(self):
        """Initialize SQLite database with optimized schema."""
        conn = sqlite3.connect(DB_PATH, check_same_thread=False) # check_same_thread=False is needed for FastAPI's async nature
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
        conn.commit()
        logging.info(f"SQLite database initialized at {DB_PATH}")
        return conn

    def _init_chroma(self):
        """Initialize ChromaDB for vector storage."""
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection_name = "one_piece_knowledge"
        try:
             collection = client.get_collection(name=collection_name)
             logging.info(f"Connected to existing ChromaDB collection: {collection_name}")
        except: # Catch exception if collection doesn't exist
             collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} # Optimize for semantic similarity
            )
             logging.info(f"Created new ChromaDB collection: {collection_name}")

        logging.info(f"ChromaDB initialized at {CHROMA_DB_PATH}")
        return client, collection

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_wiki_page(self, title):
        """Fetch a page from the One Piece wiki with improved parsing."""
        logging.debug(f"Attempting to fetch wiki page: {title}")
        url = f"https://onepiece.fandom.com/api.php?action=parse&page={title}&format=json&prop=wikitext|categories"
        try:
            response = requests.get(url, timeout=15) # Increased timeout slightly
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if "parse" not in data:
                 logging.warning(f"Could not parse wiki data for {title}")
                 return None, [], None

            wikitext = data["parse"]["wikitext"]["*"]
            parsed = mwparserfromhell.parse(wikitext)

            # Clean and extract text - Keep templates for now, focus on removing infoboxes/sidebars potentially
            # A more robust parser might be needed for complex wiki pages
            # Simple approach: remove large templates that aren't core content
            for node in parsed.ifilter_templates():
                 # Example: remove templates starting with 'Infobox' or 'Character Infobox'
                 template_name = str(node.name).strip().lower()
                 if template_name.startswith('infobox') or 'sidebar' in template_name:
                     try:
                         parsed.remove(node)
                     except ValueError:
                         pass # Node might already be removed

            # Extract internal links
            links = []
            for link in parsed.ifilter_wikilinks():
                link_title = str(link.title).split("#")[0].strip() # Remove section links
                if ":" not in link_title and len(link_title) > 1 and not link_title.startswith(('File:', 'Category:', 'Template:')):
                    links.append(link_title)

            # Determine category
            category = "Other"
            if "categories" in data["parse"]:
                categories = [cat["*"] for cat in data["parse"]["categories"]]
                for cat_type, cat_list in WIKI_CATEGORIES.items():
                    if any(cat.replace(' ', '_') in [c.replace(' ', '_') for c in categories] for cat in cat_list):
                        category = cat_type
                        break

            text = parsed.strip_code().strip() # Strip all wiki code
            text = re.sub(r'https?://\S+', '', text) # Remove URLs
            text = re.sub(r'\[\[[^\]]+\]\]', '', text) # Remove any remaining links/categories in text
            text = re.sub(r'\s+', ' ', text).strip() # Normalize all whitespace to single spaces
            text = re.sub(r'\n{2,}', '\n\n', text) # Normalize multiple newlines to double newline

            logging.debug(f"Successfully fetched and parsed {title}, category: {category}, links found: {len(links)}")
            return text, links, category

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching {title}: {e}")
            raise # Re-raise to trigger retry
        except Exception as e:
            logging.error(f"Error processing wiki data for {title}: {e}")
            return None, [], None # Return None on other errors

    def _fetch_category_pages(self, category):
        """Fetch all pages in a specific category."""
        logging.info(f"Fetching pages for category: {category}")
        url = f"https://onepiece.fandom.com/api.php?action=query&list=categorymembers&cmtitle=Category:{category}&cmlimit=500&format=json"
        try:
            response = requests.get(url, timeout=20) # Increased timeout
            response.raise_for_status()
            data = response.json()

            pages = []
            if "query" in data and "categorymembers" in data["query"]:
                for member in data["query"]["categorymembers"]:
                     # Filter out specific namespaces like "File:", "Category:", etc.
                    if member["ns"] == 0 and "title" in member:
                         pages.append(member["title"])

            logging.info(f"Found {len(pages)} pages in category {category}")
            return pages
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching category {category}: {e}")
            return []
        except Exception as e:
            logging.error(f"Error processing category {category} members: {e}")
            return []


    def _process_wiki_data(self):
        """Process wiki pages and store in database and vector store."""
        logging.info("Background processing: Starting data collection...")

        processed_count = 0

        # First process crucial pages
        logging.info("Processing crucial pages...")
        for page in CRUCIAL_PAGES:
             # Check if already processed recently
             cur = self.db_conn.execute("SELECT last_fetched FROM wiki_data WHERE title = ?", (page,))
             result = cur.fetchone()
             if result and time.time() - result[0] < REFRESH_INTERVAL:
                 logging.debug(f"Skipping recent crucial page: {page}")
                 processed_count += 1
                 continue

             if self._process_page(page):
                 processed_count += 1


        # Then fetch and process category pages
        logging.info("Processing category pages...")
        crawled_pages_from_categories = set()
        for category_type, categories in WIKI_CATEGORIES.items():
            for category in categories:
                try:
                    pages = self._fetch_category_pages(category)
                    for page in pages:
                         # Avoid processing pages already processed or in queue from crucial list
                         if page in CRUCIAL_PAGES: continue
                         if page in crawled_pages_from_categories: continue

                         crawled_pages_from_categories.add(page) # Track pages found in categories

                         # Check if already processed recently
                         cur = self.db_conn.execute("SELECT last_fetched FROM wiki_data WHERE title = ?", (page,))
                         result = cur.fetchone()
                         if result and time.time() - result[0] < REFRESH_INTERVAL:
                             logging.debug(f"Skipping recent category page: {page}")
                             processed_count += 1
                             continue

                         if self._process_page(page):
                             processed_count += 1


                except Exception as e:
                    logging.error(f"Error processing category {category}: {e}")

        self.initial_processing_done.set() # Signal that initial processing is complete
        logging.info(f"Initial data processing finished. Processed {processed_count} pages.")
        logging.info(f"Vector collection count after initial processing: {self.chroma_collection.count()}")


        # Periodically refresh data
        while True:
            time.sleep(REFRESH_INTERVAL)
            logging.info("Starting periodic refresh cycle...")

            # Refresh data for a batch of existing pages ordered by oldest fetch time
            cur = self.db_conn.execute("SELECT title FROM wiki_data ORDER BY last_fetched ASC LIMIT 200") # Increased batch size
            pages_to_refresh = [row[0] for row in cur.fetchall()]

            logging.info(f"Refreshing {len(pages_to_refresh)} pages.")
            for page in pages_to_refresh:
                 self._process_page(page) # This checks for recent fetch time internally

            logging.info("Periodic refresh cycle finished.")


    def _process_page(self, title):
        """Process a single wiki page."""
        with self.data_lock: # Use lock to prevent concurrent processing of the same page
            if title in self.processing_pages:
                logging.debug(f"Page {title} already in processing queue.")
                return False # Indicate that processing was skipped
            # Check if we *really* need to fetch this page again (double check under lock)
            cur = self.db_conn.execute("SELECT last_fetched FROM wiki_data WHERE title = ?", (title,))
            result = cur.fetchone()
            if result and time.time() - result[0] < REFRESH_INTERVAL:
                 logging.debug(f"Page {title} already processed recently under lock.")
                 return False # Indicate that processing was skipped

            self.processing_pages.add(title)
            logging.info(f"Processing page: {title}")

        try:
            # Fetch and process the page - happens outside the initial lock
            content, links, category = self._fetch_wiki_page(title)
            if not content:
                logging.warning(f"No content fetched for {title}")
                return False # Indicate failure

            # Store in SQLite (requires lock)
            with self.data_lock:
                 self.db_conn.execute(
                    "INSERT OR REPLACE INTO wiki_data VALUES (?, ?, ?, ?, ?)",
                    (title, content, category, time.time(), ','.join(links))
                 )
                 self.db_conn.commit()
                 logging.debug(f"Stored {title} in SQLite.")

            # Chunk and embed
            chunks = self._chunk_text(content, title)
            if chunks:
                try:
                    # ChromaDB operations can be done outside the main data_lock
                    # but ensure ChromaDB client/collection access is thread-safe if needed,
                    # although PersistentClient is generally safe.
                    embeddings = self.embedder.encode(chunks, convert_to_tensor=False).tolist()
                    ids = [f"{title}::{i}" for i in range(len(chunks))]
                    metadatas = [{"source": title, "category": category} for _ in chunks]

                    # Remove old chunks for this page before upserting new ones
                    try:
                        old_ids = self.chroma_collection.get(where={"source": title}, include=[])["ids"]
                        if old_ids:
                            self.chroma_collection.delete(ids=old_ids)
                            logging.debug(f"Removed {len(old_ids)} old chunks for {title} from ChromaDB.")
                    except Exception as delete_e:
                         logging.warning(f"Could not delete old chunks for {title} from ChromaDB: {delete_e}")


                    self.chroma_collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunks,
                        metadatas=metadatas
                    )
                    logging.info(f"Processed {title}: {len(chunks)} chunks added to ChromaDB.")
                except Exception as chroma_e:
                     logging.error(f"Error adding/updating chunks for {title} in ChromaDB: {chroma_e}")
                     # Decide if you want to re-raise or just log the error
                     return False # Indicate failure if vector store update fails

            # Process linked pages if needed (can be done outside the main lock)
            # Add linked pages to processing queue
            if links:
                logging.debug(f"Adding linked pages from {title} to processing queue.")
                for link in links[:10]: # Limit to first 10 links to avoid too much sprawl
                     # Queue the page processing, don't block here
                     threading.Thread(target=self._process_page, args=(link,), daemon=True).start()


            return True # Indicate success

        except Exception as e:
            logging.error(f"Caught unexpected error during processing of {title}: {e}")
            return False # Indicate failure
        finally:
            # Ensure the page is removed from processing set even on error
            with self.data_lock:
                 if title in self.processing_pages:
                    self.processing_pages.remove(title)
                    logging.debug(f"Removed {title} from processing queue.")


    def _chunk_text(self, text, title):
        """Split text into chunks for embedding with sentence awareness."""
        # Use nltk for better sentence tokenization if possible, but regex is ok for a start
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_sentences = [], []
        current_chunk_tokens = 0

        # Simple check for minimal content
        if len(sentences) < 2 and len(text.split()) < 50:
             logging.debug(f"Page {title} is too short for chunking ({len(text.split())} words). Skipping.")
             return []


        for i, sentence in enumerate(sentences):
            # Use tokenizer for token counting
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # Check if adding the sentence would exceed chunk size OR if it's the last sentence
            # If we add the sentence, what would the new token count be including a space?
            # (approximate)
            new_token_count = current_chunk_tokens + sentence_tokens + (1 if current_chunk_tokens > 0 else 0)

            if new_token_count > CHUNK_SIZE_TOKENS and current_sentences:
                # If adding sentence exceeds chunk size, save current sentences as a chunk
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text: # Ensure chunk is not empty
                    chunks.append(chunk_text)
                logging.debug(f"Chunked: {len(chunk_text.split())} words, {len(self.tokenizer.encode(chunk_text, add_special_tokens=False))} tokens")

                # Start new chunk with overlap
                overlap_sentences = current_sentences[-CHUNK_OVERLAP:] if len(current_sentences) > CHUNK_OVERLAP else current_sentences
                current_sentences = overlap_sentences
                current_chunk_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens=False)) for s in current_sentences)

            current_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens + (1 if current_chunk_tokens > 0 else 0) # Add space token count

        # Add the last chunk if any sentences remain
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text: # Ensure chunk is not empty
                 chunks.append(chunk_text)
            logging.debug(f"Final Chunked: {len(chunk_text.split())} words, {len(self.tokenizer.encode(chunk_text, add_special_tokens=False))} tokens")

        # Filter out chunks that are too short
        return [chunk for chunk in chunks if len(chunk.split()) > 20]


    def _interpret_query(self, query):
        """Interpret and expand the query using conversation history."""
        # Check if initial data processing is done before using history for complex interpretation
        if not self.initial_processing_done.is_set() or len(query.split()) > 3 and not (query.lower().startswith("and ") or query.lower().startswith("what about ")):
             # If initial data not ready, or query is already substantial, return original
             return query

        # Handle follow-up questions and vague queries using the LLM
        logging.debug(f"Interpreting query: '{query}'")
        try:
            prompt = f"""
Based on this conversation history:
{self._format_history()}

The user asked a question: "{query}"
Please interpret this as a complete, standalone question about One Piece, incorporating context from the history if necessary. Ensure the reformulated question is clear and specific, even if the original query was vague or a follow-up.
Only provide the complete reformulated question and nothing else.
"""
            # Use the generator for interpretation
            # Set max_new_tokens low as we only expect a short question
            interpretation_response = self.generator(
                 prompt,
                 max_new_tokens=50,
                 temperature=0.5,
                 do_sample=True,
                 repetition_penalty=1.1
            )[0]["generated_text"]

            # Extract the reformulated question (look for the part after the instruction)
            # This is a bit fragile, depends on LLM output format
            if "Only provide the complete reformulated question and nothing else:" in interpretation_response:
                 interpreted_query = interpretation_response.split("Only provide the complete reformulated question and nothing else:")[-1].strip()
            else:
                 # Fallback if the LLM doesn't follow the format strictly
                 # Try to clean up surrounding text
                 lines = interpreted_query.split('\n')
                 interpreted_query = lines[-1].strip() if lines else interpreted_query.strip()


            # Basic cleaning
            interpreted_query = interpreted_query.replace('"', '').strip()
            logging.info(f"Interpreted '{query}' as '{interpreted_query}'")
            return interpreted_query

        except Exception as e:
            logging.error(f"Error interpreting query '{query}': {e}. Using original query.")
            return query # Fallback to original query on error


    def _find_relevant_chunks(self, query):
        """Find relevant chunks using vector similarity."""
        # Wait briefly for initial data if not ready. Avoid long blocking.
        if not self.initial_processing_done.is_set():
             logging.warning("Initial data processing not complete. Search results may be limited.")
             # Give it a few seconds just in case it's almost done, but don't block indefinitely
             self.initial_processing_done.wait(timeout=5) # Wait up to 5 seconds

        interpreted_query = self._interpret_query(query)

        # Enhanced query expansion for key topics - performed *after* interpretation
        # This adds specific keywords to the query that might help retrieve relevant documents
        keywords_to_add = []
        lower_query = interpreted_query.lower()

        if "joy boy" in lower_query or "nika" in lower_query:
             keywords_to_add.extend(["Hito Hito no Mi Model Nika", "Sun God Nika"])
        if "blackbeard" in lower_query and "devil fruit" in lower_query:
             keywords_to_add.extend(["multiple devil fruits", "Yami Yami no Mi", "Gura Gura no Mi"])
        if "gorosei" in lower_query or "im" in lower_query:
             keywords_to_add.extend(["Five Elders", "Empty Throne", "World Government"])
        if "void century" in lower_query:
             keywords_to_add.extend(["Poneglyphs", "Ancient Kingdom", "Ohara"])


        if keywords_to_add:
            interpreted_query_with_keywords = interpreted_query + " " + " ".join(keywords_to_add)
            logging.debug(f"Expanded query: {interpreted_query_with_keywords}")
        else:
            interpreted_query_with_keywords = interpreted_query


        # Search vector database
        try:
            query_embedding = self.embedder.encode(interpreted_query_with_keywords).tolist()
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=MAX_CONTEXT_CHUNKS,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")
            return [], [] # Return empty if search fails


        chunks = []
        sources = set() # Use a set to keep sources unique

        if results and results["documents"]:
             # Only include chunks above similarity threshold
             for i, doc in enumerate(results["documents"][0]):
                 distance = results["distances"][0][i]
                 similarity = 1 - distance # Cosine distance is 1 - cosine similarity

                 logging.debug(f"Chunk {i+1}: Source={results['metadatas'][0][i]['source']}, Distance={distance:.4f}, Similarity={similarity:.4f}")

                 if similarity >= SIMILARITY_THRESHOLD:
                     chunks.append(doc)
                     sources.add(results["metadatas"][0][i]["source"])

        logging.info(f"Found {len(chunks)} relevant chunks for query '{query}'. Sources: {list(sources)}")
        return chunks, list(sources)


    def _extract_entities(self, text):
        """Extract potential One Piece entities from text."""
        # This method isn't currently used in answer_question, but kept from original.
        # Simple heuristic: look for capitalized words that might be names/places
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Filter out common English words and short words that aren't likely entities
        common_words = {"The", "A", "An", "In", "On", "At", "For", "With", "And", "But", "Or", "Not", "Is", "Are", "Was", "Were"}
        return [e for e in entities if len(e) > 3 and e not in common_words]


    def _format_history(self):
        """Format conversation history for the LLM prompt."""
        if not self.conversation_history:
            return "No recent conversation history."

        history = "Recent conversation history:\n"
        for i, (q, a) in enumerate(self.conversation_history):
            history += f"Turn {i+1}:\nUser: {q}\nAssistant: {a}\n"
        return history

    def answer_question(self, question: str):
        """Answer a question using retrieved context and conversation history."""
        logging.info(f"Received question: '{question}'")

        # Wait for initial data processing if not finished.
        # This makes the first few requests potentially slower but ensures some data is loaded.
        # Consider a proper "loading" status endpoint if this is too slow.
        if not self.initial_processing_done.is_set():
            logging.warning("Initial data processing not yet complete. Waiting up to 60s...")
            if not self.initial_processing_done.wait(timeout=60):
                logging.error("Initial data processing timed out. Cannot answer reliably.")
                return "The knowledge base is still loading. Please try again in a few minutes."
            else:
                logging.info("Initial data processing finished while waiting.")


        chunks, sources = self._find_relevant_chunks(question)

        if not chunks:
             logging.warning(f"No relevant chunks found for question: '{question}'")
             # Add a fallback or a "I don't know" response if no context is found
             fallback_prompt = f"""You are an expert on the One Piece manga and anime. The user asked: "{question}". However, no relevant specific information was found in your knowledge base. Provide a general, helpful answer based on your broad understanding of One Piece, or state that you don't have specific information on this topic. Do not invent facts.

IMPORTANT: Start immediately with your answer."""
             try:
                 response = self.generator(fallback_prompt, max_new_tokens=200, temperature=0.7, do_sample=True)[0]["generated_text"].strip()
                 # Clean up prompt instructions
                 response = re.sub(r'^.*?IMPORTANT: Start immediately with your answer\.', '', response, flags=re.DOTALL).strip()
                 answer = response
                 # Add to history (can decide if you want "I don't know" in history)
                 self.conversation_history.append((question, answer))
                 return answer
             except Exception as e:
                  logging.error(f"Error generating fallback response: {e}")
                  answer = "I couldn't find specific information about that in my knowledge base."
                  self.conversation_history.append((question, answer))
                  return answer


        # Construct the prompt for the LLM
        # Enhanced prompt with clear output instructions
        prompt = f"""You are an expert on the One Piece manga and anime. Answer the following question based *only* on the provided context and your knowledge of One Piece lore.

{self._format_history()}

Context information:
{chr(10).join(chunks)}

Question: {question}

Provide a detailed, accurate answer based on the context above. If the context doesn't contain enough information to fully answer, use your general One Piece knowledge but prioritize information from the context. Explain connections between characters and events clearly. Structure your answer logically.

IMPORTANT: Your answer must be directly useful and not include phrases like "based on the context" or "answer the question". Start immediately with your answer. Ensure your answer is cohesive and well-formatted.
"""

        logging.debug(f"Sending prompt to LLM:\n{prompt}")

        try:
            # Get the response from the generator pipeline
            response = self.generator(prompt)[0]["generated_text"]
            logging.debug(f"Raw LLM response:\n{response}")

            # Extract just the answer portion and clean it
            # Use a pattern that matches the instruction part robustly
            answer_match = re.search(r'IMPORTANT:.*?Start immediately with your answer\.(.*)', response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # Fallback extraction if the instruction pattern is not found
                # Attempt to find the question again and take everything after it
                answer_parts = response.split("Question: " + question)
                if len(answer_parts) > 1:
                     answer = answer_parts[-1].strip()
                else:
                     # If still not found, take the last significant block or the whole thing
                     logging.warning("Could not find extraction pattern, falling back to end of response.")
                     answer = response.strip() # Or implement more sophisticated fallback logic


            # Clean up any lingering prompt instructions or metadata accidentally generated
            answer = re.sub(r'^(.*?)(?:IMPORTANT:|Based on this conversation history:|Context information:|Question:)', '', answer, flags=re.DOTALL | re.IGNORECASE).strip()
            answer = re.sub(r'\s*Sources:\s*.*$', '', answer, flags=re.DOTALL) # Remove any "Sources:" line the LLM might invent


        except Exception as e:
            logging.error(f"Error generating response from LLM: {e}")
            answer = "Sorry, I encountered an error while generating the response."


        # Add to conversation history (user question and generated answer)
        self.conversation_history.append((question, answer))
        logging.debug(f"Added to history: Q='{question}', A='{answer[:50]}...'")

        # Format response with better source attribution
        if sources:
            # Ensure sources are clean titles, not links or complex text
            clean_sources = [s.replace('_', ' ') for s in sources] # Simple cleaning
            sources_list = list(clean_sources)[:5] # Limit to top 5 sources for brevity in output
            sources_str = ", ".join(sources_list)
            # Append sources clearly
            return f"{answer}\n\nSources: {sources_str}"

        return answer

# --- FastAPI Application Setup ---

# Initialize the chatbot instance globally
# This happens when the script is first imported/run, loading models and starting threads
try:
    chatbot = OnePieceChatbot()
    logging.info("OnePieceChatbot instance created.")
except Exception as e:
    logging.critical(f"Failed to initialize OnePieceChatbot: {e}")
    # In a real deployment, this might stop the container or signal an error
    # For now, we'll let the app start but requests will likely fail

app = FastAPI()

# Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "One Piece Chatbot API is running. Send a POST request to /ask with your question."}

# Health check endpoint
@app.get("/health")
async def health_check():
    # Add more sophisticated checks here if needed (e.g., db connection, model loaded)
    if not hasattr(chatbot, 'generator') or chatbot.generator is None:
         return {"status": "error", "message": "LLM model not loaded"}, 500

    # Check if the background thread has completed initial processing
    if not chatbot.initial_processing_done.is_set():
         return {"status": "warning", "message": "Initial data processing still in progress. Some answers may be limited."}, 200 # Or 503 Service Unavailable

    # Optional: Check ChromaDB count to ensure it's not empty after initialization
    try:
         count = chatbot.chroma_collection.count()
         if count == 0:
             return {"status": "warning", "message": "Knowledge base is empty after initialization. Data fetching might have failed."}, 200
         return {"status": "ok", "message": "Chatbot is ready.", "knowledge_base_size": count}, 200
    except Exception as e:
         logging.error(f"Health check failed during ChromaDB count: {e}")
         return {"status": "warning", "message": f"Health check encountered an issue: {e}"}, 200


# Endpoint to answer questions
@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    """
    Submit a question about One Piece and get an answer.
    """
    question = request.question
    if not question or not question.strip():
        return {"answer": "Please provide a question."}

    # Run the synchronous answer_question method in a thread pool
    # This prevents blocking the FastAPI event loop
    try:
        answer = await run_in_threadpool(chatbot.answer_question, question)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error processing question '{question}': {e}")
        return {"answer": "Sorry, an internal error occurred while processing your question."}, 500 # Return 500 status code on error

# To run this application:
# 1. Save the code as a Python file (e.g., main.py).
# 2. Make sure you have the necessary libraries installed:
#    pip install fastapi uvicorn transformers sentence-transformers tenacity mwparserfromhell chromadb requests sqlite3 accelerate bitsandbytes torch huggingface_hub
#    (You might need additional dependencies for torch/accelerate based on your hardware, e.g., CUDA)
# 3. Run from your terminal: uvicorn main:app --reload
# For deployment, you'll typically use a production server like Gunicorn with Uvicorn workers:
#    gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
# Ensure the HF_TOKEN environment variable is set in your deployment environment.