# 선행 연구

형식: 제목 | 날짜 | 우리와의 차이 한 줄.

## Recurrent-state 양자화 (직접 경쟁)
- DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization | 2026.08 | GDN/KDA state 단독 PTQ 최초 주장. 27B 이상. KV와의 joint 없음. arXiv 2608.27513
- LeapQuant: Efficient Linear Attention with ... recurrent-state quantization | 2026.09 | 8bit state에서 near-lossless, compensator tokens. state 단독. arXiv 2609.38166
- STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization | 2026.09 | lifetime-aware bit allocation, state 내부 배분. KV 미고려. arXiv 2609.38169
- Why Gated DeltaNet Survives 4-Bit Quantization (NVFP4 W4A4) | 2026.09 | GDN 레이어 weight/activation 양자화. state 캐시 자체는 아님. arXiv 2609.04098

## KV cache 양자화 (배경)
- KIVI (2024), KVQuant (NeurIPS 2024), QAQ, GEAR, SQuat, MiniKV, VecInfer, RateQuant (2026), InnerQ (2026) | 전부 순수 어텐션 모델. state 없음.
- PM-KVQ (2025.05), ThinKV (2025.10), KITTY (2025.11), MixKVQ (ACL 2026) | long-CoT 추론 모델용 KV 양자화. 하이브리드 아님.
- Quantization Hurts Reasoning? (2025.04) | 추론 모델 양자화 실증 연구. KV 4bit lossless, 3bit 급락. 하이브리드 아님.

## 인접 (선점 확인용)
- QuantMLA | 2026.09 | MLA dual-path 양자화. 아키텍처 다름.
- Quant-dLLM (ICLR 2026), DLLMQuant / FAIR-Calib (ICML 2026), STaR-Quant (2026.06) | diffusion LLM 양자화. 아키텍처 다름.
- QuantVLA (CVPR 2026) | VLA 양자화.
