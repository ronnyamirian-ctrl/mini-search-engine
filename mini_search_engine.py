"""
Mini Search Engine in Python 3.14.3

Features:
- Web crawler
- Inverted index
- BM25 ranking
- Title boost
- Phrase boost
- Snippet generation
- Simple Flask API

Install:
    pip install requests beautifulsoup4 flask

Run:
    python mini_search_engine.py

Search API:
    http://127.0.0.1:5000/search?q=your+query
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from collections import defaultdict, Counter, deque
from typing import Dict, List, Set, Tuple, Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify


# -----------------------------
# Text processing
# -----------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "if", "in", "into", "is", "it", "its", "no", "not", "of", "on", "or",
    "such", "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with", "you", "your", "i", "we", "he", "she",
    "them", "our", "who", "what", "when", "where", "why", "how", "which",
    "have", "has", "had", "been", "were", "do", "does", "did", "so", "too",
    "can", "could", "should", "would", "may", "might", "must"
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase_tokens: List[str], doc_tokens: List[str]) -> bool:
    if not phrase_tokens or len(phrase_tokens) > len(doc_tokens):
        return False
    n = len(phrase_tokens)
    for i in range(len(doc_tokens) - n + 1):
        if doc_tokens[i:i + n] == phrase_tokens:
            return True
    return False


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Document:
    doc_id: int
    url: str
    title: str
    text: str
    tokens: List[str] = field(default_factory=list)
    length: int = 0


# -----------------------------
# Search engine
# -----------------------------

class MiniSearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: Dict[int, Document] = {}
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_lengths: Dict[int, int] = {}
        self.total_docs = 0
        self.avg_doc_len = 0.0

    def add_document(self, url: str, title: str, text: str) -> int:
        doc_id = len(self.documents)
        title = normalize_space(title) if title else url
        text = normalize_space(text)
        tokens = tokenize(title + " " + text)

        doc = Document(
            doc_id=doc_id,
            url=url,
            title=title,
            text=text,
            tokens=tokens,
            length=len(tokens),
        )

        self.documents[doc_id] = doc
        self.doc_lengths[doc_id] = doc.length
        self.total_docs += 1

        tf = Counter(tokens)
        for term, freq in tf.items():
            self.inverted_index[term][doc_id] = freq
            self.doc_freq[term] += 1

        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, self.total_docs)
        return doc_id

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

    def _bm25(self, query_terms: List[str], doc_id: int) -> float:
        doc = self.documents[doc_id]
        doc_len = max(1, doc.length)
        score = 0.0

        for term in query_terms:
            postings = self.inverted_index.get(term)
            if not postings or doc_id not in postings:
                continue

            tf = postings[doc_id]
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
            score += idf * (tf * (self.k1 + 1)) / denom

        return score

    def _title_boost(self, query_terms: List[str], doc: Document) -> float:
        title_tokens = tokenize(doc.title)
        if not title_tokens:
            return 0.0

        query_set = set(query_terms)
        title_set = set(title_tokens)
        overlap = len(query_set & title_set)
        return 0.4 * overlap

    def _phrase_boost(self, query_terms: List[str], doc: Document) -> float:
        if len(query_terms) < 2:
            return 0.0
        if phrase_in_text(query_terms, doc.tokens):
            return 1.2
        return 0.0

    def _url_boost(self, query_terms: List[str], doc: Document) -> float:
        url_text = tokenize(doc.url)
        if not url_text:
            return 0.0
        overlap = len(set(query_terms) & set(url_text))
        return 0.15 * overlap

    def _make_snippet(self, doc: Document, query_terms: List[str], max_len: int = 220) -> str:
        text = doc.text
        lowered = text.lower()

        positions = []
        for term in query_terms:
            pos = lowered.find(term)
            if pos != -1:
                positions.append(pos)

        if positions:
            start = max(0, min(positions) - 80)
        else:
            start = 0

        snippet = text[start:start + max_len]
        snippet = normalize_space(snippet)

        if start > 0:
            snippet = "..." + snippet
        if start + max_len < len(text):
            snippet += "..."
        return snippet

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query = normalize_space(query)
        query_terms = tokenize(query)

        if not query_terms:
            return []

        candidate_docs: Set[int] = set()
        for term in query_terms:
            candidate_docs.update(self.inverted_index.get(term, {}).keys())

        results = []
        for doc_id in candidate_docs:
            doc = self.documents[doc_id]

            score = self._bm25(query_terms, doc_id)
            score += self._title_boost(query_terms, doc)
            score += self._phrase_boost(query_terms, doc)
            score += self._url_boost(query_terms, doc)

            if score > 0:
                results.append((score, doc_id))

        results.sort(reverse=True, key=lambda x: x[0])

        output = []
        for score, doc_id in results[:top_k]:
            doc = self.documents[doc_id]
            output.append({
                "doc_id": doc_id,
                "title": doc.title,
                "url": doc.url,
                "score": round(score, 4),
                "snippet": self._make_snippet(doc, query_terms),
            })
        return output


# -----------------------------
# Web crawling
# -----------------------------

class Crawler:
    def __init__(
        self,
        engine: MiniSearchEngine,
        max_pages: int = 100,
        max_depth: int = 2,
        timeout: int = 10,
        delay: float = 0.5,
        same_domain_only: bool = True,
        user_agent: str = "MiniSearchBot/1.0",
    ):
        self.engine = engine
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        self.delay = delay
        self.same_domain_only = same_domain_only
        self.headers = {"User-Agent": user_agent}
        self.visited: Set[str] = set()

    def crawl(self, seed_urls: List[str]) -> None:
        queue = deque()
        for url in seed_urls:
            queue.append((url, 0))

        pages_crawled = 0

        while queue and pages_crawled < self.max_pages:
            url, depth = queue.popleft()
            if url in self.visited or depth > self.max_depth:
                continue

            self.visited.add(url)

            try:
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    continue

                html = resp.text
                soup = BeautifulSoup(html, "html.parser")

                title = ""
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()

                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                text = soup.get_text(" ", strip=True)
                text = normalize_space(text)

                if len(text) < 80:
                    continue

                self.engine.add_document(url=url, title=title, text=text)
                pages_crawled += 1
                print(f"Indexed {pages_crawled}: {url}")

                base_domain = urlparse(url).netloc

                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    abs_url = urljoin(url, href)
                    parsed = urlparse(abs_url)

                    if parsed.scheme not in {"http", "https"}:
                        continue
                    if self.same_domain_only and parsed.netloc != base_domain:
                        continue
                    if abs_url not in self.visited:
                        queue.append((abs_url, depth + 1))

                time.sleep(self.delay)

            except Exception as e:
                print(f"Failed to crawl {url}: {e}")


# -----------------------------
# Demo data
# -----------------------------

def load_demo_documents(engine: MiniSearchEngine) -> None:
    docs = [
        (
            "local://1",
            "Python Search Engine Tutorial",
            "This tutorial explains how to build a search engine in Python using BM25, crawling, and inverted indexes.",
        ),
        (
            "local://2",
            "Information Retrieval Basics",
            "Information retrieval systems rank documents using scoring models like TF-IDF, BM25, and link analysis.",
        ),
        (
            "local://3",
            "Flask Web API Guide",
            "Flask is a lightweight web framework for building APIs and web applications in Python.",
        ),
        (
            "local://4",
            "Web Crawling Basics",
            "A crawler visits web pages, extracts links, and collects content for indexing and search.",
        ),
        (
            "local://5",
            "Ranking Search Results",
            "Ranking combines relevance signals, title matching, phrase matching, and document quality scores.",
        ),
    ]

    for url, title, text in docs:
        engine.add_document(url=url, title=title, text=text)


# -----------------------------
# Flask API
# -----------------------------

engine = MiniSearchEngine()
load_demo_documents(engine)

app = Flask(__name__)


@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Mini Search</title>

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

body{
    font-family:Arial,sans-serif;
    background:white;
    min-height:100vh;
}

.topbar{
    display:flex;
    justify-content:flex-end;
    align-items:center;
    gap:20px;
    padding:15px 25px;
}

.topbar a{
    text-decoration:none;
    color:#202124;
    font-size:14px;
}

.topbar a:hover{
    text-decoration:underline;
}

.profile{
    width:34px;
    height:34px;
    border-radius:50%;
    background:#4285f4;
}

.center{
    display:flex;
    flex-direction:column;
    align-items:center;
    margin-top:120px;
}

.logo{
    font-size:92px;
    font-weight:500;
    user-select:none;
    margin-bottom:30px;
}

.blue{color:#4285F4;}
.red{color:#EA4335;}
.yellow{color:#FBBC05;}
.green{color:#34A853;}

.search-box{
    width:580px;
    max-width:90%;
    height:48px;
    border:1px solid #dfe1e5;
    border-radius:24px;
    padding:0 20px;
    font-size:16px;
    outline:none;
}

.search-box:hover{
    box-shadow:0 1px 6px rgba(32,33,36,.28);
}

.search-box:focus{
    box-shadow:0 1px 6px rgba(32,33,36,.28);
}

.buttons{
    margin-top:25px;
}

button{
    border:none;
    background:#f8f9fa;
    padding:10px 20px;
    margin:5px;
    border-radius:4px;
    cursor:pointer;
}

button:hover{
    border:1px solid #dadce0;
}

#results{
    width:70%;
    margin:40px auto;
    text-align:left;
}

.result{
    margin-bottom:30px;
}

.result-title{
    color:#1a0dab;
    font-size:22px;
    margin-bottom:5px;
}

.result-url{
    color:green;
    font-size:14px;
    margin-bottom:5px;
}

.result-snippet{
    color:#4d5156;
}

</style>
</head>

<body>

<div class="topbar">
    <a href="#">Gmail</a>
    <a href="#">Images</a>
    <div class="profile"></div>
</div>

<div class="center">

    <div class="logo">
        <span class="blue">G</span>
        <span class="red">o</span>
        <span class="yellow">o</span>
        <span class="blue">g</span>
        <span class="green">l</span>
        <span class="red">e</span>
    </div>

    <input
        class="search-box"
        id="q"
        placeholder="Search..."
        autocomplete="off"
    >

    <div class="buttons">
        <button onclick="search()">Search</button>
    </div>

</div>

<div id="results"></div>

<script>

async function search(){

    let q =
        document.getElementById("q").value;

    let response =
        await fetch(`/search?q=${encodeURIComponent(q)}`);

    let data =
        await response.json();

    let html = "";

    data.results.forEach(r => {

        html += `
        <div class="result">
            <div class="result-title">
                ${r.title}
            </div>

            <div class="result-url">
                ${r.url}
            </div>

            <div class="result-snippet">
                ${r.snippet}
            </div>
        </div>
        `;
    });

    document.getElementById("results")
        .innerHTML = html;
}

document.getElementById("q")
.addEventListener("keypress", function(e){

    if(e.key === "Enter"){
        search();
    }
});

</script>

</body>
</html>
"""




@app.get("/search")
def search_route():
    q = request.args.get("q", "").strip()
    k = request.args.get("k", 10, type=int)
    results = engine.search(q, top_k=k)
    return jsonify({
        "query": q,
        "count": len(results),
        "results": results,
    })


@app.post("/add")
def add_route():
    data = request.get_json(force=True, silent=False)
    url = data.get("url", "")
    title = data.get("title", "")
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "text is required"}), 400

    doc_id = engine.add_document(url=url, title=title, text=text)
    return jsonify({"ok": True, "doc_id": doc_id})


@app.post("/crawl")
def crawl_route():
    data = request.get_json(force=True, silent=False)
    seeds = data.get("seeds", [])
    max_pages = int(data.get("max_pages", 20))
    max_depth = int(data.get("max_depth", 1))

    if not isinstance(seeds, list) or not seeds:
        return jsonify({"error": "seeds must be a non-empty list"}), 400

    crawler = Crawler(
        engine=engine,
        max_pages=max_pages,
        max_depth=max_depth,
        delay=0.3,
        same_domain_only=True,
    )
    crawler.crawl(seeds)
    return jsonify({"ok": True, "indexed_docs": len(engine.documents)})


# -----------------------------
# CLI
# -----------------------------

def cli_search_loop() -> None:
    print("Mini Search Engine CLI")
    print("Type a query and press Enter. Type 'exit' to quit.\n")
    while True:
        query = input("search> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        results = engine.search(query, top_k=5)
        if not results:
            print("No results.\n")
            continue

        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}")
            print(f"   score={r['score']}")
            print(f"   {r['snippet']}\n")


if __name__ == "__main__":
    print("Mini Search Engine started.")
    print("Open http://127.0.0.1:5000")
    if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5000)
     app.run(debug=True)
     
