# Reference verification (arXiv API, 2026-10-15)

All 26 arXiv identifiers cited in the draft resolve, and every title matches the cited work. `paper/refs_verified.bib`
holds bib entries generated from the arXiv metadata (full author lists; venues filled where known, otherwise arXiv
preprint). Four non-arXiv entries (Qwen3.5 blog, vLLM issue, fla repository, Seide 2014) are appended by hand.

Corrections for the manuscript text:
- [23] LeapQuant: the actual title is "LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization"
  (the draft wrote "near-lossless recurrent-state quantization").
- [4] Erbas, Intes and Pandey (2609.04490) is a GRU encoder-decoder for fluorescence lifetime imaging, not an LLM
  study; its abstract mentions GRU only. The sentence "a study of GRU and LSTM write-back explains why coarse state
  storage freezes a recurrence and how error feedback repairs it" must be checked against the full text before
  keeping "LSTM" and "error feedback"; safest wording: "a study of recurrent-state write-back in a GRU model from
  a different domain [4] isolates the same freezing effect".
- "all but the last evaluate models of 27B parameters or more": supported for DAMP (35B, 48B in the abstract),
  STEPQuant (27B, 48B) and the NVFP4 study (Qwen3.8-27B); LeapQuant's and DASC's abstracts do not state sizes,
  so either check their experiments sections or soften to "most of these works evaluate models of 27B parameters
  or more".
- DAMP's abstract supports the introduction's claim that INT8/FP8 state storage degrades reasoning and INT4
  collapses it.
- KVTuner is arXiv:2502.04420 (Xing Li et al., 2025); RateQuant is arXiv:2605.06675 (Fei Zuo et al., 2026).

| key | arXiv | year | first author (n) | title |
| dong2024qaq | 2403.04643 | 2024 | Shichen Dong (4) | QAQ: Quality Adaptive Quantization for LLM KV Cache |
| frantar2023gptq | 2210.17323 | 2022 | Elias Frantar (4) | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers |
| schlag2021fwp | 2102.11174 | 2021 | Imanol Schlag (3) | Linear Transformers Are Secretly Fast Weight Programmers |
| karimireddy2019ef | 1901.09847 | 2019 | Sai Praneeth Karimireddy (4) | Error Feedback Fixes SignSGD and other Gradient Compression Schemes |
| dao2024mamba2 | 2405.21060 | 2024 | Tri Dao (2) | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality |
| ashkboos2024quarot | 2404.00456 | 2024 | Saleh Ashkboos (9) | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs |
| yang2024deltanet | 2406.06484 | 2024 | Songlin Yang (5) | Parallelizing Linear Transformers with the Delta Rule over Sequence Length |
| kim2026qad | 2607.04244 | 2026 | Jaeyeon Kim (3) | Quantize the Target, Quantize the Drafter: Efficient Inference with Qwen3.5-4B |
| liu2024kivi | 2402.02750 | 2024 | Zirui Liu (8) | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache |
| saxena2025kvlinc | 2510.05373 | 2025 | Utkarsh Saxena (2) | KVLinC : KV Cache Quantization with Hadamard Rotation and Linear Correction |
| lin2024awq | 2306.00978 | 2023 | Ji Lin (10) | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration |
| liu2025pmkvq | 2505.18610 | 2025 | Tengxuan Liu (10) | PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs |
| erbas2026writeback | 2609.04490 | 2026 | Ismail Erbas (3) | When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference |
| zhang2026damp | 2608.27513 | 2026 | Tao Zhang (9) | DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization |
| pan2026leapquant | 2609.38166 | 2026 | Yi Pan (13) | LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization |
| yu2026dasc | 2608.30386 | 2026 | Yanqi Yu (7) | DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving |
| liu2025quanthurts | 2504.04823 | 2025 | Ruikang Liu (8) | Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models |
| hooper2024kvquant | 2401.18079 | 2024 | Coleman Hooper (7) | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization |
| kimi2025linear | 2510.26692 | 2025 |  Kimi Team (60) | Kimi Linear: An Expressive, Efficient Attention Architecture |
| yang2025gdn | 2412.06464 | 2024 | Songlin Yang (3) | Gated Delta Networks: Improving Mamba2 with Delta Rule |
| zhang2025mixkvq | 2512.19206 | 2025 | Tao Zhang (5) | MixKVQ: Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning |
| kang2024gear | 2403.05527 | 2024 | Hao Kang (7) | GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM |
| kozyrev2026nvfp4 | 2609.04098 | 2026 | Sergii Kozyrev (2) | Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM |
| yao2026stepquant | 2609.38169 | 2026 | Bingchen Yao (9) | STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization |
| li2025kvtuner | 2502.04420 | 2025 | Xing Li (9) | KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference |
| zuo2026ratequant | 2605.06675 | 2026 | Fei Zuo (5) | RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory |
