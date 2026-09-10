"""
download_papers.py — Download open-access full texts from PubMed Central
Uses the PMC OA API to get full article text, saves as .txt files for ingestion.
Usage: python download_papers.py
"""

import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

PAPERS_DIR = Path("papers")
PAPERS_DIR.mkdir(exist_ok=True)

BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL  = "vetlit-rag"
EMAIL = "daliabelaid@gmail.com"

SEARCHES = [
    ("dairy cattle productive longevity survival analysis prediction",   3),
    ("mastitis detection dairy cow somatic cell count machine learning", 3),
    ("dairy cattle welfare body condition score lameness",               2),
    ("bovine disease prediction random forest XGBoost",                  2),
]


def search_pmc(query, max_results):
    r = requests.get(f"{BASE}/esearch.fcgi", params={
        "db": "pmc", "term": query + " AND open access[filter]",
        "retmax": max_results, "retmode": "json",
        "tool": TOOL, "email": EMAIL,
    }, timeout=15)
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch_full_text(pmcid):
    """Fetch full article XML from PMC OA service."""
    r = requests.get(f"{BASE}/efetch.fcgi", params={
        "db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml",
        "tool": TOOL, "email": EMAIL,
    }, timeout=30)
    if r.status_code != 200:
        return None, None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None, None

    # Extract title
    title_el = root.find(".//article-title")
    title = (title_el.text or "").strip() if title_el is not None else f"PMC{pmcid}"
    title = title[:80]

    # Extract all body text
    parts = []

    # Abstract
    for abs_el in root.findall(".//abstract"):
        for p in abs_el.iter("p"):
            text = "".join(p.itertext()).strip()
            if text:
                parts.append(text)

    # Body sections
    for sec in root.findall(".//body//sec"):
        title_el = sec.find("title")
        if title_el is not None:
            sec_title = "".join(title_el.itertext()).strip()
            if sec_title:
                parts.append(f"\n## {sec_title}\n")
        for p in sec.findall("p"):
            text = "".join(p.itertext()).strip()
            if text:
                parts.append(text)

    # Fallback: grab all <p> tags if body was empty
    if not parts:
        for p in root.iter("p"):
            text = "".join(p.itertext()).strip()
            if text and len(text) > 50:
                parts.append(text)

    full_text = "\n\n".join(parts)
    return title, full_text


if __name__ == "__main__":
    print("\n=== PubMed Central Full-Text Downloader ===\n")
    saved = 0
    seen = set()

    for query, n in SEARCHES:
        print(f"Searching: {query[:60]}...")
        ids = search_pmc(query, n)
        print(f"  Found: {ids}")

        for pmcid in ids:
            if pmcid in seen:
                continue
            seen.add(pmcid)

            print(f"  Fetching PMC{pmcid}...", end=" ", flush=True)
            title, text = fetch_full_text(pmcid)

            if not text or len(text) < 500:
                print("skipped (no full text available)")
                time.sleep(0.4)
                continue

            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in (title or pmcid)).strip()
            fname = f"PMC{pmcid}_{safe_title[:45]}.txt"
            out = PAPERS_DIR / fname
            out.write_text(f"Title: {title}\nPMCID: PMC{pmcid}\n\n{text}", encoding="utf-8")
            print(f"saved ({len(text)//1000}k chars) -> {fname}")
            saved += 1
            time.sleep(0.4)

    print(f"\nDone - {saved} articles saved to ./papers/")
    if saved == 0:
        print("Nothing downloaded. Check your internet connection.")
