# 02 — Download & validate embedding model

**What to build:** Download `paraphrase-multilingual-MiniLM-L12-v2` and confirm it produces 384-dim embeddings that capture semantic similarity between acronyms and their expansions — the exact failure mode this rebuild addresses.

**Blocked by:** 01 (Install deps & validate GPU)

**Status:** ready-for-agent

## Detailed instructions

1. Run this Python script (GPU must be available from ticket 01):
   ```python
   from sentence_transformers import SentenceTransformer
   import numpy as np

   model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')

   test_emb = model.encode(
       ["ICICI Bank Mumbai", "Industrial Credit and Investment Corporation of India Bank"],
       convert_to_numpy=True
   )
   print(f"Shape: {test_emb.shape}")  # expect (2, 384)

   cos = np.dot(test_emb[0], test_emb[1]) / (np.linalg.norm(test_emb[0]) * np.linalg.norm(test_emb[1]))
   print(f"Cosine similarity: {cos:.4f}")  # expect noticeably > 0.3
   ```

2. The model will be cached to `~/.cache/huggingface/` (~470MB). Confirm the download completes.

3. Verify:
   - Output shape is `(2, 384)`.
   - Cosine similarity is noticeably above 0.3 (these two strings share almost no character n-grams but are semantically the same entity — this is why we need embeddings).

No code files are created in this ticket — just model download and validation. The model is loaded fresh in each downstream script.

- [x] Model downloads successfully (~470MB)
- [x] `test_emb.shape == (2, 384)`
- [x] Cosine similarity between "ICICI Bank Mumbai" and full expansion is > 0.3
- [x] No CUDA OOM error during encoding
