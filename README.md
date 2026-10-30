## Hi, I'm Bhaskar Gurram

I'm an AI/ML engineer at Zasti Inc. I mostly work on the performance and numerical correctness of scientific and ML software, usually by replacing a slow or fragile computation with a better algorithm and testing it against a reference implementation.

[Website](https://bhaskar.website) · [Google Scholar](https://scholar.google.com/citations?user=0tSEbKIAAAAJ) · [LinkedIn](https://www.linkedin.com/in/bhaskar-gurram) · gurrambhaskar.ai@gmail.com

### Open-source work I'm most happy with

- **JAX**: [batched `lexsort` / `unique(axis=...)`](https://github.com/jax-ml/jax/pull/41186), so compile time no longer grows with the number of sort keys (`lexsort` on 100 keys: 162 s to 0.22 s). Also [`gammaincinv`](https://github.com/jax-ml/jax/pull/41512), the inverse incomplete gamma function, tested against mpmath.
- **statsmodels**: [closed-form leave-one-out influence measures](https://github.com/statsmodels/statsmodels/pull/10367) (35 s to 2.7 ms at n = 10,000, matching R), [lag-order selection from one QR](https://github.com/statsmodels/statsmodels/pull/10571), and [the Zivot-Andrews test from one QR](https://github.com/statsmodels/statsmodels/pull/10602) (checked against R's `urca`).
- **xarray**: [`interp` skips a redundant sort](https://github.com/pydata/xarray/pull/11658) on already-sorted coordinates (164 ms to 2.9 ms).
- **Apache Arrow (arrow-rs)**: [shared Arrow-to-Parquet conversions in the Parquet writer](https://github.com/apache/arrow-rs/pull/11741), including a fix for how the dictionary encoder's hash lookup was compiled.
- **Hugging Face candle**: [row tiling in the CPU quantized matmul](https://github.com/huggingface/candle/pull/4003), about 1.45x faster prefill with identical outputs.
- **Rust `cc` crate**: [stop passing rustc's PGO flags to the C compiler](https://github.com/rust-lang/cc-rs/pull/2003), which fixed segfaulting PGO builds.

All my pull requests are [here](https://github.com/pulls?q=is%3Apr+author%3Abhaskargurram-ai).

### Research

- **Valid Per-Field Selective Risk Control for Document Extraction.** [arXiv:2608.14639](https://arxiv.org/abs/2608.14639), 2026. Code: [verifydoc](https://github.com/bhaskargurram-ai/verifydoc).
- **Auditing Automated Evaluation, Error Propagation, and Runtime Mitigation in Tool-Using Language Agents.** [arXiv:2604.16706](https://arxiv.org/abs/2604.16706), 2026. Code: [agenthallu-bench](https://github.com/bhaskargurram-ai/agenthallu-bench).

### Projects

- [**verifydoc**](https://github.com/bhaskargurram-ai/verifydoc): calibrated per-field confidence and source grounding for document-to-JSON extraction.
- [**unwind**](https://github.com/bhaskargurram-ai/unwind): an undo log for agent tool calls, as an MCP proxy.
- [**HARNESS-DB**](https://github.com/harness-db/harness-db): a coded dataset of 1,256 agent harnesses.

### Background

M.S. in Computer Science, University of Cincinnati (2024). B.Tech. in Computer Science, SRM Institute of Science and Technology (2022).
