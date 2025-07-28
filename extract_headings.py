"""
extract_headings.py

Extract a semantic document title plus H1/H2/H3 headings from a PDF
using an offline Sentence‑Transformer model and hierarchical clustering.
Outputs a flat JSON "outline" with page numbers.
"""

import re
import json
import argparse

import fitz  # PyMuPDF
import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def extract_pages(pdf_path):
    """Open the PDF and return list of raw page‑texts."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def normalize_text(text):
    """
    Flatten hyphenated words and newlines so paragraphs stay intact.
    """
    text = text.replace("-\n", "")    # join hyphen‑broken words
    text = text.replace("\n", " ")
    return text


def split_paragraphs(text):
    """
    Split on two or more spaces (real paragraph breaks),
    and drop any very-short noise.
    """
    paras = re.split(r"\s{2,}", text)
    return [p.strip() for p in paras if len(p.strip()) > 20]


def chunk_paragraphs(paragraphs_with_pages, overlap=1):
    """
    Given a list of (paragraph, page_index), produce:
      - chunks:      list of chunk‑texts
      - chunk_pages: list of page_index for each chunk (from the first para in that chunk)
    """
    chunks, chunk_pages = [], []
    for i in range(len(paragraphs_with_pages)):
        group = paragraphs_with_pages[i : i + overlap + 1]
        text = " ".join(p for p, _ in group)
        page = group[0][1]
        chunks.append(text)
        chunk_pages.append(page)
    return chunks, chunk_pages


def embed_chunks(chunks, model_path):
    """Encode all chunks to embeddings using the local model."""
    model = SentenceTransformer(model_path)
    return model.encode(chunks, show_progress_bar=True)


def select_representative_idx(idxs, embeddings):
    """
    Given a list of original indices and the full embeddings array,
    pick the index whose embedding is closest to the centroid.
    """
    sub_embeds = embeddings[idxs]
    centroid = np.mean(sub_embeds, axis=0)
    dists = np.linalg.norm(sub_embeds - centroid, axis=1)
    best_sub = int(np.argmin(dists))
    return idxs[best_sub]


def cluster_indices(idxs, embeddings, n_clusters):
    """
    Agglomerative clustering over the embeddings of the given idxs.
    Returns a dict: cluster_label -> list of original indices.
    """
    if len(idxs) < n_clusters:
        return {0: idxs}

    sub_embeds = embeddings[idxs]
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(sub_embeds)

    clusters = {}
    for lbl in set(labels):
        clusters[lbl] = [idxs[i] for i, l in enumerate(labels) if l == lbl]
    return clusters


def build_outline(chunks, embeddings, chunk_pages, h1, h2, h3):
    """
    Returns a flat list of {"level","text","page"} entries,
    walking H1 → H2 → H3.
    """
    outline = []

    # Top‑level doc title
    all_idxs = list(range(len(chunks)))
    title_idx = select_representative_idx(all_idxs, embeddings)
    # We set this aside in main()

    # Level 1
    lvl1 = cluster_indices(all_idxs, embeddings, h1)
    for lbl1 in sorted(lvl1):
        idxs1 = lvl1[lbl1]
        rep1 = select_representative_idx(idxs1, embeddings)
        outline.append({
            "level": "H1",
            "text": chunks[rep1],
            "page": chunk_pages[rep1],
        })

        # Level 2
        lvl2 = cluster_indices(idxs1, embeddings, h2)
        for lbl2 in sorted(lvl2):
            idxs2 = lvl2[lbl2]
            rep2 = select_representative_idx(idxs2, embeddings)
            outline.append({
                "level": "H2",
                "text": chunks[rep2],
                "page": chunk_pages[rep2],
            })

            # Level 3
            lvl3 = cluster_indices(idxs2, embeddings, h3)
            for lbl3 in sorted(lvl3):
                idxs3 = lvl3[lbl3]
                rep3 = select_representative_idx(idxs3, embeddings)
                outline.append({
                    "level": "H3",
                    "text": chunks[rep3],
                    "page": chunk_pages[rep3],
                })

    return title_idx, outline


def main():
    p = argparse.ArgumentParser(
        description="Extract semantic title + H1/H2/H3 headings from a PDF"
    )
    p.add_argument("pdf", help="Path to input PDF file")
    p.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="Path to local Sentence‑Transformer model directory",
    )
    p.add_argument(
        "--h1", type=int, default=5, help="Number of top‑level (H1) clusters"
    )
    p.add_argument(
        "--h2", type=int, default=3, help="Number of second‑level (H2) clusters per H1"
    )
    p.add_argument(
        "--h3", type=int, default=2, help="Number of third‑level (H3) clusters per H2"
    )
    p.add_argument(
        "--out", default="headings.json", help="Output JSON file path"
    )
    args = p.parse_args()

    print("1/6 • Extracting raw pages…")
    pages = extract_pages(args.pdf)

    print("2/6 • Splitting into paragraphs with page tags…")
    paras_with_pages = []
    for pg_i, text in enumerate(pages):
        norm = normalize_text(text)
        for para in split_paragraphs(norm):
            paras_with_pages.append((para, pg_i))
    print(f"   • Found {len(paras_with_pages)} paragraphs.")

    print("3/6 • Creating overlapping chunks…")
    chunks, chunk_pages = chunk_paragraphs(paras_with_pages, overlap=1)
    print(f"   • Created {len(chunks)} chunks.")

    print("4/6 • Embedding chunks…")
    embeddings = embed_chunks(chunks, args.model)

    print("5/6 • Clustering & selecting title + headings…")
    title_idx, outline = build_outline(
        chunks, embeddings, chunk_pages, args.h1, args.h2, args.h3
    )
    doc_title = chunks[title_idx]

    output = {
        "title": doc_title,
        "outline": outline
    }

    print(f"6/6 • Saving output to {args.out} …")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✅ Done!")


if __name__ == "__main__":
    main()

