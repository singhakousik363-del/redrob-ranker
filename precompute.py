#!/usr/bin/env python3
"""
Offline pre-computation - run ONCE before rank.py
Saves JD embedding + title embeddings to disk
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading BGE model...", flush=True)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# JD ka essence - jo actually matter karta hai
JD_TEXT = """
Senior AI Engineer founding team. Production experience with embeddings, 
vector search, retrieval systems, ranking, recommendation engines. 
Built and shipped ML systems at scale. Experience with FAISS, Elasticsearch, 
vector databases, semantic search, dense retrieval, re-ranking, learning to rank.
Python expert. NLP, transformers, BERT, LLMs. Search quality evaluation NDCG MRR.
5-9 years experience. Product company experience preferred.
"""

print("Embedding JD...", flush=True)
jd_embedding = model.encode(JD_TEXT, normalize_embeddings=True)
np.save("jd_embedding.npy", jd_embedding)
print(f"JD embedding saved: shape {jd_embedding.shape}")

# Ab 100K candidates ke titles + summaries embed karo
print("Loading candidates...", flush=True)
candidate_ids = []
texts = []

with open("candidates.jsonl") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        c = json.loads(line)
        cid = c["candidate_id"]
        
        # Title + current role + summary combine karo
        title = c["profile"].get("current_title", "")
        headline = c["profile"].get("headline", "")
        summary = c["profile"].get("summary", "")[:300]
        
        # Last 2 career descriptions bhi add karo
        career_text = " ".join(
            ch.get("title", "") + " " + ch.get("description", "")[:150]
            for ch in c["career_history"][:2]
        )
        
        combined = f"{title}. {headline}. {summary}. {career_text}"
        
        candidate_ids.append(cid)
        texts.append(combined)
        
        if (i+1) % 10000 == 0:
            print(f"  Prepared {i+1}/100000 texts", flush=True)

print(f"\nEncoding {len(texts)} candidates in batches...", flush=True)
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Embeddings shape: {embeddings.shape}")

# Save karo
np.save("candidate_embeddings.npy", embeddings)
np.save("candidate_ids.npy", np.array(candidate_ids))

print("\nDone! Files saved:")
print(f"  jd_embedding.npy - {jd_embedding.nbytes/1024:.1f} KB")
print(f"  candidate_embeddings.npy - {embeddings.nbytes/1024/1024:.1f} MB")
print(f"  candidate_ids.npy")
print("\nAb rank.py run karo - embeddings use honge automatically")
