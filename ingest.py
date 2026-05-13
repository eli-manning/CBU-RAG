"""
CBU Knowledge Base Ingestion
Chunks and embeds CBU content into ChromaDB.

Usage:
    python ingest.py --url https://www.calbaptist.edu/academics/
    python ingest.py --seed          # scrape all default CBU URLs
    python ingest.py --pdf ./docs/handbook.pdf
    python ingest.py --dir ./docs/
    python ingest.py --reset         # wipe collection and start fresh
"""

import argparse
import hashlib
import time
from pathlib import Path

import chromadb
import ollama
import requests
from bs4 import BeautifulSoup

# --- Config ---
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

CBU_SEED_URLS = [
    "https://www.calbaptist.edu/academics/",
    "https://www.calbaptist.edu/admissions/",
    "https://calbaptist.edu/life-at-cbu/",
    "https://www.calbaptist.edu/about/",
    "https://www.calbaptist.edu/engineering/",
    "https://calbaptist.edu/academics/programs/bachelor-of-science-computer-science/",
    "https://calbaptist.edu/academics/programs/minor-computer-science",
    "https://calbaptist.edu/academics/programs/minor-computer-engineering",
    "https://calbaptist.edu/academics/programs/master-of-science-computer-science/",
    "https://calbaptist.edu/academics/programs/minor-data-sciences",
    "https://calbaptist.edu/academics/programs/bachelor-of-science-data-science/",
    "https://calbaptist.edu/academics/programs/bachelor-of-science-electrical-and-computer-engineering/",
    "https://calbaptist.edu/academics/programs/minor-software-engineering-and-app-development"
]

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(
    name="cbu_knowledge",
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk.strip()) < 50:
            continue
        chunks.append({"text": chunk, "source": source})
    return chunks


def embed_and_store(chunks: list[dict]):
    for chunk in chunks:
        doc_id = hashlib.md5(chunk["text"].encode()).hexdigest()
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"])["embedding"]
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )
    print(f"  Stored {len(chunks)} chunks.")


def scrape_url(url: str):
    print(f"Scraping: {url}")
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "CBU-Bot/1.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        chunks = chunk_text(text, source=url)
        embed_and_store(chunks)
        time.sleep(0.5)
    except Exception as e:
        print(f"  Error: {e}")


def ingest_pdf(pdf_path: str):
    try:
        import pymupdf
        print(f"Ingesting PDF: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
        chunks = chunk_text(text, source=pdf_path)
        embed_and_store(chunks)
    except ImportError:
        print("Install pymupdf: pip install pymupdf")


def ingest_directory(dir_path: str):
    for path in Path(dir_path).rglob("*"):
        if path.suffix == ".txt":
            print(f"Ingesting: {path}")
            text = path.read_text(errors="ignore")
            embed_and_store(chunk_text(text, str(path)))
        elif path.suffix == ".pdf":
            ingest_pdf(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Single URL to scrape")
    parser.add_argument("--seed", action="store_true", help="Scrape all seed URLs")
    parser.add_argument("--pdf", help="PDF to ingest")
    parser.add_argument("--dir", help="Directory of docs to ingest")
    parser.add_argument("--reset", action="store_true", help="Wipe collection and exit")
    args = parser.parse_args()

    if args.reset:
        client.delete_collection("cbu_knowledge")
        print("Collection wiped.")
    elif args.url:
        scrape_url(args.url)
    elif args.seed:
        for url in CBU_SEED_URLS:
            scrape_url(url)
    elif args.pdf:
        ingest_pdf(args.pdf)
    elif args.dir:
        ingest_directory(args.dir)
    else:
        parser.print_help()

    print(f"Total docs in collection: {collection.count()}")
