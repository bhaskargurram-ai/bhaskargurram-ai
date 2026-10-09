## Bhaskar Gurram

AI/ML engineer at Zasti Inc. (Ashburn, VA). M.S. in Computer Science, University of Cincinnati.

I work on the performance and numerical correctness of scientific and ML software. Most of my open-source work is algorithmic speedups and correctness fixes in libraries that other people build on.

[Website](https://bhaskar.website) · [Google Scholar](https://scholar.google.com/citations?user=0tSEbKIAAAAJ) · [LinkedIn](https://www.linkedin.com/in/bhaskar-gurram) · [Email](mailto:gurrambhaskar.ai@gmail.com)

---

### Merged upstream contributions

| Project | Contribution | Result |
|---|---|---|
| **Google JAX** | [Batched `lexsort` / `unique(axis=...)`](https://github.com/jax-ml/jax/pull/41186): new `batch_size` keyword, designed with a core maintainer over six review rounds | Compile time no longer grows with the number of keys: `jnp.unique` 15.5 s → 0.28 s; `lexsort` on 100 keys 162 s → 0.22 s |
| **statsmodels** | [Closed-form leave-one-out influence measures](https://github.com/statsmodels/statsmodels/pull/10367) (`dfbetas`, `dffits`, `cov_ratio`, studentized residuals); closes maintainer issue #9009 | 35 s → 2.7 ms at n = 10,000; matches R's `influence.measures` to 12 decimals |
| **xarray** | [`interp` skips `sortby` on already-sorted coordinates](https://github.com/pydata/xarray/pull/11658) | Vectorized interp 164 ms → 2.9 ms |
| **Hugging Face candle** | [Row tiling in the CPU quantized matmul for prefill](https://github.com/huggingface/candle/pull/4003) | About 1.45x faster prefill for Q4K / Q8_0; outputs bit-identical |
| **Bokeh** | [Row/Column layout reserves room for aligned borders](https://github.com/bokeh/bokeh/pull/15487) (BokehJS) | Fixes cropped axes and legends; milestone 4.0, backported |
| **Apple MLX (mlx-lm)** | [Left-padding mask in batched generation](https://github.com/ml-explore/mlx-lm/pull/1925) | Fixes wrong logits for left-padded batches in two model families |
| **Google JAX** | [`zeta` JVP rule](https://github.com/jax-ml/jax/pull/41017), [`pareto` support boundary](https://github.com/jax-ml/jax/pull/41015) | Correct values and gradients outside the domain and at the support edge |
| **LangSmith SDK** | [Keep the HTTP response on `raise_for_status_with_text` errors](https://github.com/langchain-ai/langsmith-sdk/pull/3597) | Shipped in v0.14.2 |
| **CISA CSET** (US DHS) | [Raise the PBKDF2 work factor to OWASP guidance](https://github.com/cisagov/cset/pull/5611) | Password-hash hardening in CISA's Cyber Security Evaluation Tool |
| **USGS dataretrieval** (US DOI) | [Open-ended (`..`) date ranges in the OGC client](https://github.com/DOI-USGS/dataretrieval-python/pull/440) | Fixes silently unfiltered results and HTTP 400s in the official USGS water-data client |
| **Sandia National Laboratories pyGSTi** | [Default POVM for circuits on a subset of a model's qubits](https://github.com/sandialabs/pyGSTi/pull/932); fixes #721, a 0.11 release blocker | Circuits on part of a multi-qubit model no longer fail with "Missing POVM"; merged by a Sandia maintainer |
| **ogx** | [Dependency floor for the inline provider](https://github.com/ogx-ai/ogx/pull/6669) | Fixes a user-reported install failure |
| **cartography (CNCF)** | [Handle empty Google Workspace groups](https://github.com/cartography-cncf/cartography/pull/2018) | Fixes a sync crash |

---

### Research

- **Valid Per-Field Selective Risk Control for Document Extraction: Three Failure Modes, a Validity Ladder, and When Conditioning Pays.** B. Gurram. [arXiv:2608.14639](https://arxiv.org/abs/2608.14639), 2026. Code: [verifydoc](https://github.com/bhaskargurram-ai/verifydoc).
- **Auditing Automated Evaluation, Error Propagation, and Runtime Mitigation in Tool-Using Language Agents.** B. Gurram. [arXiv:2604.16706](https://arxiv.org/abs/2604.16706), 2026. Code: [agenthallu-bench](https://github.com/bhaskargurram-ai/agenthallu-bench).
- **Agentic systematic literature review: a multi-agent LLM pipeline evaluated on 62 randomized controlled trials across five medical specialties.** Submitted to *Expert Systems with Applications*, 2026. Code: [agentic-slr](https://github.com/bhaskargurram-ai/agentic-slr).
- **M.S. thesis (2024):** EEG-fMRI fusion with temporal convolutional networks for decoding visual stimuli. Accuracy is 84.8% within subject and 81.1% leave-one-subject-out, compared with 65.5% for EEG only and 74.6% for fMRI only. Advisor: Prof. Vikram Ravindra.

### Projects

- [**verifydoc**](https://github.com/bhaskargurram-ai/verifydoc): calibrated per-field confidence and source grounding for document-to-JSON extraction, with accept/review abstention.
- [**unwind**](https://github.com/bhaskargurram-ai/unwind): a reversibility layer for agent tool calls. It is an MCP proxy with a cross-server undo log.
- [**provio**](https://github.com/writ-agent/provio): authorization and provenance for AI agent tool calls.
- [**HARNESS-DB**](https://github.com/harness-db/harness-db): a coded dataset of 1,256 agent harnesses.

---

### Experience

- **AI Engineer, Zasti Inc.** (2024–present): agentic and retrieval systems for clinical research. Retrieval over 50M+ document embeddings, and 4-bit GPTQ deployment that cut model size by 75%.
- **Graduate Teaching Assistant, University of Cincinnati** (2022–2024): Python, cloud and ML labs for 500+ students.
- **Deep Learning Developer, Technocolabs Softwares** (2021–2022): led a four-person team building medical pattern-recognition models.

### Education

- **M.S. Computer Science**, University of Cincinnati, 2024. GPA 3.95/4.0. Graduate Incentive Award.
- **B.Tech. Computer Science**, SRM Institute of Science and Technology, 2022. GPA 3.98/4.0.
- **MicroMasters in Data Science**, UC San Diego, 2022.
