"""
Fetches paper abstracts (+ optionally full text via arXiv's PDF, if you
add a PDF-text-extraction step) from the arXiv API for a set of search
queries, and writes them out as a corpus.json the pipeline can ingest.

No API key required -- arXiv's API is free and public. Only needs
internet access, which is why this is a separate script you run locally
rather than something baked into the offline test suite.

Usage:
    python data/fetch_corpus.py --queries "retrieval augmented generation" \
        "hallucination detection LLM" "dense passage retrieval" \
        --max-per-query 40 --out data/corpus.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_query(query: str, max_results: int = 40) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.find("atom:id", NS).text.strip().split("/")[-1]
        title = entry.find("atom:title", NS).text.strip()
        title = re.sub(r"\s+", " ", title)
        summary = entry.find("atom:summary", NS).text.strip()
        summary = re.sub(r"\s+", " ", summary)
        published = entry.find("atom:published", NS).text.strip()
        authors = [a.find("atom:name", NS).text
                   for a in entry.findall("atom:author", NS)]
        papers.append({
            "doc_id": arxiv_id,
            "title": title,
            "text": f"{title}\n\n{summary}",
            "published": published,
            "authors": authors,
            "source_query": query,
        })
    return papers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", required=True,
                         help="Search queries, e.g. 'retrieval augmented generation'")
    parser.add_argument("--max-per-query", type=int, default=40)
    parser.add_argument("--out", default="data/corpus.json")
    parser.add_argument("--sleep", type=float, default=3.0,
                         help="Seconds between API calls (be a polite arXiv citizen)")
    args = parser.parse_args()

    all_papers = {}
    for i, q in enumerate(args.queries):
        print(f"Fetching '{q}' ({args.max_per_query} results)...")
        try:
            papers = fetch_query(q, max_results=args.max_per_query)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        for p in papers:
            all_papers[p["doc_id"]] = p  # dedupe across queries
        print(f"  Got {len(papers)} papers (total unique so far: {len(all_papers)})")
        if i < len(args.queries) - 1:
            time.sleep(args.sleep)

    with open(args.out, "w") as f:
        json.dump(list(all_papers.values()), f, indent=2)
    print(f"\nSaved {len(all_papers)} unique papers to {args.out}")


if __name__ == "__main__":
    main()
