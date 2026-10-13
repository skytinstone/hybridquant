# 선행 연구

형식은 제목 | 날짜 | 우리와의 차이 한 줄입니다.

## Recurrent-state 양자화 (직접 경쟁)
- DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization | 2026.08 | GDN/KDA state만 PTQ한 최초 연구라고 주장합니다. 27B 이상 모델을 다루며 KV와의 joint allocation은 없습니다. arXiv 2608.27513
- LeapQuant: Efficient Linear Attention with ... recurrent-state quantization | 2026.09 | 8bit state에서 near-lossless이고 compensator tokens를 씁니다. state만 다룹니다. arXiv 2609.38166
- STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization | 2026.09 | lifetime-aware bit allocation으로 state 내부에서 비트를 배분합니다. KV는 고려하지 않습니다. arXiv 2609.38169
- Why Gated DeltaNet Survives 4-Bit Quantization (NVFP4 W4A4) | 2026.09 | GDN 레이어의 weight/activation 양자화입니다. state 캐시 자체는 다루지 않습니다. arXiv 2609.04098

## KV cache 양자화 (배경)
- KIVI (2024), KVQuant (NeurIPS 2024), QAQ, GEAR, SQuat, MiniKV, VecInfer, RateQuant (2026), InnerQ (2026) | 모두 순수 어텐션 모델 대상이며 state가 없습니다.
- PM-KVQ (2025.05), ThinKV (2025.10), KITTY (2025.11), MixKVQ (ACL 2026) | long-CoT 추론 모델용 KV 양자화입니다. 하이브리드가 아닙니다.
- Quantization Hurts Reasoning? (2025.04) | 추론 모델 양자화 실증 연구입니다. KV 4bit는 lossless, 3bit에서 급락합니다. 하이브리드가 아닙니다.

## 인접 (선점 확인용)
- QuantMLA | 2026.09 | MLA dual-path 양자화입니다. 아키텍처가 다릅니다.
- Quant-dLLM (ICLR 2026), DLLMQuant / FAIR-Calib (ICML 2026), STaR-Quant (2026.06) | diffusion LLM 양자화입니다. 아키텍처가 다릅니다.
- QuantVLA (CVPR 2026) | VLA 양자화입니다.
