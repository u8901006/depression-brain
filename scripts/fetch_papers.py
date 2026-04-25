#!/usr/bin/env python3
"""
Fetch latest TRD (Treatment-Resistant Depression) research papers from PubMed.
Keywords and journals based on trd_journals_pubmed_templates.md.
"""

import json
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.parse import quote_plus

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

JOURNALS = [
    "American Journal of Psychiatry",
    "JAMA Psychiatry",
    "The Lancet Psychiatry",
    "Biological Psychiatry",
    "Molecular Psychiatry",
    "Neuropsychopharmacology",
    "Psychological Medicine",
    "Journal of Affective Disorders",
    "Depression and Anxiety",
    "Journal of Psychiatric Research",
    "Progress in Neuro-Psychopharmacology and Biological Psychiatry",
    "European Neuropsychopharmacology",
    "Psychotherapy and Psychosomatics",
    "European Psychiatry",
    "World Journal of Biological Psychiatry",
    "Psychiatry Research",
    "CNS Spectrums",
    "Journal of Clinical Psychiatry",
    "Journal of Psychiatry and Neuroscience",
    "Clinical Psychology Review",
]

HEADERS = {"User-Agent": "TRDBrainBot/1.0 (research aggregator)"}


def build_query(days: int = 7, max_journals: int = 12) -> str:
    trd_core = (
        '("treatment-resistant depression"[tiab] OR '
        '"treatment resistant depression"[tiab] OR '
        "TRD[tiab] OR "
        '"refractory depression"[tiab] OR '
        '"difficult-to-treat depression"[tiab] OR '
        '"drug-resistant depression"[tiab])'
    )
    journal_part = " OR ".join(
        [f'"{j}"[Journal]' for j in JOURNALS[:max_journals]]
    )
    lookback = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y/%m/%d"
    )
    date_part = f'"{lookback}"[Date - Publication] : "3000"[Date - Publication]'
    return f"{trd_core} AND ({journal_part}) AND {date_part}"


def search_papers(query: str, retmax: int = 50) -> list[str]:
    params = (
        f"?db=pubmed&term={quote_plus(query)}"
        f"&retmax={retmax}&sort=date&retmode=json"
    )
    url = PUBMED_SEARCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"[ERROR] PubMed search failed: {e}", file=sys.stderr)
        return []


def fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    ids = ",".join(pmids)
    params = f"?db=pubmed&id={ids}&retmode=xml"
    url = PUBMED_FETCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=60) as resp:
            xml_data = resp.read().decode()
    except Exception as e:
        print(f"[ERROR] PubMed fetch failed: {e}", file=sys.stderr)
        return []

    papers = []
    try:
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            medline = article.find(".//MedlineCitation")
            art = medline.find(".//Article") if medline else None
            if art is None:
                continue

            title_el = art.find(".//ArticleTitle")
            title = (
                (title_el.text or "").strip()
                if title_el is not None and title_el.text
                else ""
            )

            abstract_parts = []
            for abs_el in art.findall(".//Abstract/AbstractText"):
                label = abs_el.get("Label", "")
                text = "".join(abs_el.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)[:2000]

            journal_el = art.find(".//Journal/Title")
            journal = (
                (journal_el.text or "").strip()
                if journal_el is not None and journal_el.text
                else ""
            )

            pub_date = art.find(".//PubDate")
            date_str = ""
            if pub_date is not None:
                year = pub_date.findtext("Year", "")
                month = pub_date.findtext("Month", "")
                day = pub_date.findtext("Day", "")
                parts = [p for p in [year, month, day] if p]
                date_str = " ".join(parts)

            pmid_el = medline.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            keywords = []
            for kw in medline.findall(".//KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())

            papers.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "date": date_str,
                    "abstract": abstract,
                    "url": link,
                    "keywords": keywords,
                }
            )
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}", file=sys.stderr)

    return papers


def main():
    parser = argparse.ArgumentParser(description="Fetch TRD papers from PubMed")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--output", default="-")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    query = build_query(days=args.days)
    print(
        f"[INFO] Searching PubMed for TRD papers (last {args.days} days)...",
        file=sys.stderr,
    )

    pmids = search_papers(query, retmax=args.max_papers)
    print(f"[INFO] Found {len(pmids)} papers", file=sys.stderr)

    if not pmids:
        print("NO_CONTENT", file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    {
                        "date": datetime.now(
                            timezone(timedelta(hours=8))
                        ).strftime("%Y-%m-%d"),
                        "count": 0,
                        "papers": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    papers = fetch_details(pmids)
    print(f"[INFO] Fetched details for {len(papers)} papers", file=sys.stderr)

    output_data = {
        "date": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d"),
        "count": len(papers),
        "papers": papers,
    }

    out_str = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output == "-":
        print(out_str)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"[INFO] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
