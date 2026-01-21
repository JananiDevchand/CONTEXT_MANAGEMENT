# context_compressor.py

import re
import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer


# =========================
# MODELS
# =========================

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

ABSTRACTIVE_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
_abstractive = None
_abstractive_tokenizer = None


def _load_abstractive():
    global _abstractive, _abstractive_tokenizer
    if _abstractive is None:
        _abstractive = pipeline("summarization", model=ABSTRACTIVE_MODEL_NAME)
        _abstractive_tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    return _abstractive, _abstractive_tokenizer


# =========================
# CONFIG
# =========================

LONG_THRESHOLD = 600               # characters
SIGNIFICANCE_THRESHOLD = 0.55
QUALITY_SIM_THRESHOLD = 0.70

EXTRACTIVE_MAX_SENTENCES_HIGH = 3
EXTRACTIVE_MAX_SENTENCES_LOW = 1

ABSTRACTIVE_MAX_LEN_HIGH = 160
ABSTRACTIVE_MAX_LEN_LOW = 80


# =========================
# HELPERS
# =========================

def _split_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def _embed(texts):
    return EMBED_MODEL.encode(texts)


# =========================
# EXTRACTIVE
# =========================

def extractive_summary(text: str, max_sentences: int):
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()

    sent_emb = _embed(sentences)
    doc_emb = _embed([text])[0]

    scores = cosine_similarity(sent_emb, doc_emb.reshape(1, -1)).flatten()
    top_idx = np.argsort(scores)[-max_sentences:]
    top_idx = sorted(top_idx)

    return " ".join(sentences[i] for i in top_idx)


# =========================
# ABSTRACTIVE
# =========================

def abstractive_summary(text: str, max_len: int):
    summarizer, tokenizer = _load_abstractive()

    tokens = tokenizer.encode(text, truncation=True, max_length=512)
    safe_text = tokenizer.decode(tokens, skip_special_tokens=True)

    result = summarizer(
        safe_text,
        max_length=max_len,
        min_length=max_len // 2,
        do_sample=False
    )

    return result[0]["summary_text"]


# =========================
# MAIN COMPRESSOR
# =========================

def compress_context(
    contexts: List[Dict],
    user_query: str,
    target_model_max_tokens: int = 4096
):
    """
    Advanced adaptive context compressor.
    """

    print("\n🧠 CONTEXT COMPRESSION PIPELINE STARTED")
    print(f"Target model max tokens: {target_model_max_tokens}")

    query_emb = _embed([user_query])[0]

    compressed_chunks = []
    debug = []

    for idx, ctx in enumerate(contexts, 1):
        original = ctx["content"]
        chunk_emb = _embed([original])[0]

        # ---------------------------------
        # 1️⃣ Semantic Significance
        # ---------------------------------
        significance = float(cosine_similarity(
            chunk_emb.reshape(1, -1),
            query_emb.reshape(1, -1)
        )[0][0])

        print(f"\n[CHUNK {idx}]")
        print(f"Semantic significance: {significance:.3f}")

        important = significance >= SIGNIFICANCE_THRESHOLD

        # ---------------------------------
        # 2️⃣ Compression aggressiveness
        # ---------------------------------
        if target_model_max_tokens >= 16000:
            extractive_sentences = EXTRACTIVE_MAX_SENTENCES_HIGH
            abstractive_len = ABSTRACTIVE_MAX_LEN_HIGH
        else:
            extractive_sentences = EXTRACTIVE_MAX_SENTENCES_LOW
            abstractive_len = ABSTRACTIVE_MAX_LEN_LOW

        # ---------------------------------
        # 3️⃣ Mode selection
        # ---------------------------------
        if len(original) > LONG_THRESHOLD:
            mode = "abstractive"
        else:
            mode = "extractive"

        print(f"Compression mode: {mode}")

        if important:
            print("Importance: HIGH → lighter compression")
        else:
            print("Importance: LOW → stronger compression")

        # ---------------------------------
        # 4️⃣ Initial compression
        # ---------------------------------
        if mode == "extractive":
            summary = extractive_summary(
                original,
                max_sentences=extractive_sentences if important else 1
            )
        else:
            summary = abstractive_summary(
                original,
                max_len=abstractive_len if important else abstractive_len // 2
            )

        # ---------------------------------
        # 5️⃣ Quality feedback loop
        # ---------------------------------
        summary_emb = _embed([summary])[0]
        quality = float(cosine_similarity(
            chunk_emb.reshape(1, -1),
            summary_emb.reshape(1, -1)
        )[0][0])

        print(f"Compression quality similarity: {quality:.3f}")

        if quality < QUALITY_SIM_THRESHOLD:
            print("⚠️ Quality too low → recompressing (lighter)")
            summary = extractive_summary(original, max_sentences=3)

        # ---------------------------------
        # Store
        # ---------------------------------
        compressed_chunks.append({
            "role": "system",
            "content": summary,
            "metadata": {
                **ctx.get("metadata", {}),
                "significance": significance,
                "compression_mode": mode,
                "quality_score": quality
            }
        })

        debug.append({
            "chunk": idx,
            "significance": significance,
            "mode": mode,
            "quality": quality,
            "original_len": len(original),
            "compressed_len": len(summary)
        })

    # ---------------------------------
    # Final join
    # ---------------------------------
    final_text = "\n\n".join(c["content"] for c in compressed_chunks)

    print("\n🧠 FINAL COMPRESSED CONTEXT")
    print("─" * 60)
    print(final_text)
    print("─" * 60)

    return [{
        "role": "system",
        "content": final_text
    }], debug
