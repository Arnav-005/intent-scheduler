"""
Stage 2 — Semantic Embedding Pipeline
Adds 32-dim PCA-compressed sentence embeddings to the notification dataset.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ── Config ────────────────────────────────────────
CSV_IN  = "notifications_10k.csv"
CSV_OUT = "notifications_10k_with_embeddings.csv"
PCA_OUT = "pca_model.pkl"        # save PCA so Stage 3 can reuse it at inference
N_COMPONENTS = 32
BATCH_SIZE = 256                 # adjust down if you hit RAM issues
SEED = 42
# ─────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(CSV_IN)
texts = df["notification_text"].tolist()
print(f"  {len(texts)} notifications loaded")

# ── 1. Encode with sentence-transformer ──────────
print("\nLoading sentence-transformer model (downloads ~80MB on first run)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Encoding {len(texts)} texts in batches of {BATCH_SIZE}...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2-normalize — good for cosine similarity
)
print(f"  Raw embedding shape: {embeddings.shape}")  # (10000, 384)

# ── 2. PCA compression 384 → 32 dims ─────────────
print(f"\nFitting PCA: 384 → {N_COMPONENTS} dimensions...")
pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
embeddings_32 = pca.fit_transform(embeddings)

explained = pca.explained_variance_ratio_.sum()
print(f"  Variance explained by {N_COMPONENTS} components: {explained:.1%}")
print(f"  Compressed embedding shape: {embeddings_32.shape}")

# ── 3. Attach to dataframe ────────────────────────
print("\nAttaching embeddings to dataframe...")
emb_cols = [f"sem_{i:02d}" for i in range(N_COMPONENTS)]
emb_df = pd.DataFrame(embeddings_32, columns=emb_cols)
df_out = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

# ── 4. Save outputs ───────────────────────────────
df_out.to_csv(CSV_OUT, index=False)
print(f"\n✓ Augmented dataset saved → {CSV_OUT}")
print(f"  Shape: {df_out.shape}")
print(f"  New columns: sem_00 ... sem_{N_COMPONENTS-1:02d}")

with open(PCA_OUT, "wb") as f:
    pickle.dump(pca, f)
print(f"✓ PCA model saved → {PCA_OUT}  (needed for Stage 3 inference)")

# ── 5. Sanity check ───────────────────────────────
print("\n── Sanity Check ─────────────────────────────────")
print("Sample semantic vectors (first 5 rows, first 5 dims):")
print(df_out[emb_cols[:5]].head())

print("\nSemantic distance check (should differ by content):")
spam_mask    = df_out["content_type"] == "spam"
academic_mask = df_out["content_type"].isin(["deadline_reminder", "assignment"])
v_spam     = embeddings_32[spam_mask].mean(axis=0)
v_academic = embeddings_32[academic_mask].mean(axis=0)
cosine_sim = np.dot(v_spam, v_academic) / (
    np.linalg.norm(v_spam) * np.linalg.norm(v_academic)
)
print(f"  Cosine similarity (spam vs academic): {cosine_sim:.3f}")
print(f"  (Lower = more semantically distinct — good for the bandit to learn from)")

print("\nAll done. Files ready for Stage 3:")
print(f"  {CSV_OUT}")
print(f"  {PCA_OUT}")