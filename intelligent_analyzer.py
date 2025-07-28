"""
intelligent_analyzer.py

Given:
  - A directory of related PDFs (3–10 files)
  - A Persona description
  - A concrete Job-to-be-Done

This script extracts all semantic chunks from each PDF, ranks them
by relevance to the Persona+Job, and writes out the top K sections.
"""
import os
import re
import json
import argparse
import numpy as np
import fitz  # PyMuPDF
from nltk import download
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK tokenizer is available
download("punkt", quiet=True)


def extract_pages(pdf_path):
    """Open the PDF and return list of raw page-texts."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def normalize_text(text):
    """
    Flatten hyphenated words and newlines so paragraphs stay intact.
    """
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    return text


def split_paragraphs(text):
    """
    Split on two or more spaces (real paragraph breaks),
    and drop any very-short noise.
    """
    paras = re.split(r"\s{2,}", text)
    return [p.strip() for p in paras if len(p.strip()) > 30]


def chunk_paragraphs(paras_with_meta, overlap=1):
    """
    Given list of {'doc', 'page', 'text'} dicts, produce:
      - chunks: list of chunk-texts
      - meta:   list of {'doc','page'} for each chunk (from the first para)
    """
    chunks = []
    meta = []
    for i in range(len(paras_with_meta)):
        group = paras_with_meta[i : i + overlap + 1]
        text = " ".join(p['text'] for p in group)
        info = { 'doc': group[0]['doc'], 'page': group[0]['page'] }
        chunks.append(text)
        meta.append(info)
    return chunks, meta


def embed_texts(texts, model_path):
    """Encode a list of texts into embeddings using the local model."""
    model = SentenceTransformer(model_path)
    return model.encode(texts, show_progress_bar=True)


def rank_chunks(chunks_meta, chunk_embeds, query_emb, top_k):
    """
    Compute cosine similarity vs query_emb, return top_k ranked sections.
    """
    sims = cosine_similarity(chunk_embeds, query_emb.reshape(1, -1)).flatten()
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idxs:
        entry = {
            'doc':   chunks_meta[i]['doc'],
            'page':  chunks_meta[i]['page'],
            'score': float(sims[i]),
            'text':  chunks_meta[i]['text'][:500]
        }
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Document Analyst"
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder with 3–10 PDFs to analyze"
    )
    parser.add_argument(
        "--persona", required=True,
        help="Short description of the persona"
    )
    parser.add_argument(
        "--job", required=True,
        help="Concrete Job-to-be-Done for that persona"
    )
    parser.add_argument(
        "--model", default="models/all-MiniLM-L6-v2",
        help="Local Sentence-Transformer model path"
    )
    parser.add_argument(
        "--topk", type=int, default=10,
        help="How many top sections to return"
    )
    parser.add_argument(
        "--out", default="analysis.json",
        help="Output JSON file"
    )
    args = parser.parse_args()

    # 1) Read & preprocess all paragraphs from each PDF
    paras = []
    files = sorted([f for f in os.listdir(args.input) if f.lower().endswith('.pdf')])
    for filename in files:
        path = os.path.join(args.input, filename)
        pages = extract_pages(path)
        for i, raw in enumerate(pages):
            norm = normalize_text(raw)
            for para in split_paragraphs(norm):
                paras.append({ 'doc': filename, 'page': i+1, 'text': para })

    if not paras:
        print("No paragraphs found in input folder.")
        return

    # 2) Chunk paragraphs (with overlap)
    chunks, meta = chunk_paragraphs(paras, overlap=1)

    # 3) Embed chunks
    print(f"Embedding {len(chunks)} chunks...")
    embeds = embed_texts(chunks, args.model)

    # 4) Build and embed persona+job query
    query = f"Persona: {args.persona}. Task: {args.job}."
    print(f"Embedding query: {query}")
    query_emb = embed_texts([query], args.model)[0]

    # 5) Rank and select top K
    print(f"Ranking top {args.topk} sections...")
    top_sections = rank_chunks(
        [{'doc': m['doc'], 'page': m['page'], 'text': chunks[i]} for i, m in enumerate(meta)],
        embeds,
        query_emb,
        args.topk
    )

    # 6) Write output
    output = {
        'persona': args.persona,
        'job':     args.job,
        'query':   query,
        'results': top_sections
    }
    with open(args.out, 'w', encoding='utf-8') as out_f:
        json.dump(output, out_f, indent=2, ensure_ascii=False)

    print(f"Analysis written to {args.out}")


if __name__ == '__main__':
    main()
