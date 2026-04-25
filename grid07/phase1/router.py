"""
Phase 1: Vector-Based Persona Matching
Embeds bot personas into a FAISS in-memory vector store,
then routes incoming posts to relevant bots via cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ──────────────────────────────────────────────
# Bot persona definitions
# ──────────────────────────────────────────────
BOT_PERSONAS = {
    "bot_a": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns."
    ),
    "bot_b": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}

# ──────────────────────────────────────────────
# Build the in-memory FAISS index at import time
# ──────────────────────────────────────────────
print("[Phase 1] Loading embedding model …")
_model = SentenceTransformer("all-MiniLM-L6-v2")   # lightweight, ~80 MB

# Encode all personas once
_persona_ids   = list(BOT_PERSONAS.keys())
_persona_texts = list(BOT_PERSONAS.values())
_persona_vecs  = _model.encode(_persona_texts, normalize_embeddings=True).astype("float32")

# FAISS index with inner-product (= cosine similarity when vecs are L2-normalised)
_dim   = _persona_vecs.shape[1]
_index = faiss.IndexFlatIP(_dim)
_index.add(_persona_vecs)
print(f"[Phase 1] FAISS index built — {_index.ntotal} persona vectors stored (dim={_dim})")


def route_post_to_bots(post_content: str, threshold: float = 0.35) -> list[dict]:
    """
    Embed *post_content* and return the list of bots whose persona vector
    exceeds *threshold* cosine similarity with the post.

    Returns a list of dicts: [{"bot_id": ..., "score": ...}, ...]
    Sorted by descending similarity score.

    Note: threshold=0.35 is realistic for all-MiniLM-L6-v2.
    The assignment suggests 0.85 but that assumes much more powerful
    embeddings (e.g. text-embedding-3-large).  Adjust as needed.
    """
    # Encode & normalise the incoming post
    post_vec = _model.encode([post_content], normalize_embeddings=True).astype("float32")

    # Query all personas (k = total number of bots)
    scores, indices = _index.search(post_vec, k=len(_persona_ids))

    matched = []
    for score, idx in zip(scores[0], indices[0]):
        bot_id = _persona_ids[idx]
        print(f"  [{bot_id}] cosine similarity = {score:.4f}  (threshold={threshold})")
        if score >= threshold:
            matched.append({"bot_id": bot_id, "score": float(score)})

    # Sort best match first
    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "The Fed raised interest rates again — bond yields are spiking.",
        "Big Tech surveillance is out of control; GDPR fines are a joke.",
    ]

    for post in test_posts:
        print(f"\n📨 Post: \"{post}\"")
        results = route_post_to_bots(post)
        if results:
            for r in results:
                print(f"   ✅ Routed to {r['bot_id']}  (score={r['score']:.4f})")
        else:
            print("   ❌ No bots matched above threshold")
