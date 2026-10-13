# 실험 로그

형식: 날짜 | 모델 | 설정 | 결과 | 결론. 한 실험 한 줄. 결과 파일은 results/ 에 jsonl로 보관.

| 날짜 | 모델 | 설정 | 결과 | 결론 |
|---|---|---|---|---|
| 2026-10-13 | (없음) | 레포/훅/평가 코드 뼈대 작성 | CPU 스모크 테스트 통과 여부는 아래 기록 | M1 시작 |
| 2026-10-13 | tiny Qwen3.5 (random init, hidden 64, 4 layers, CPU) | tests/test_smoke_cpu.py, transformers 5.19.0.dev0 | CPU 스모크 테스트 통과 (12 passed): 훅 발화, 레이어 인덱스 일치, (16,16) vs (2,2) decode NLL 상이, 패치 원복 | M1 로컬 검증 완료, 다음은 Colab에서 Qwen3.5-0.8B |
| 2026-10-13 | Qwen3.5-0.8B (MPS, M3 Pro, bf16) | scripts/check_hooks.py, KV2/S4, prefill 128, decode 32 | hook coverage PASS: layers_seen_kv=[3,7,11,15,19,23], layers_seen_state=나머지 18개, use_kernels=False, 모델 로드 포함 70s | M1 항목 1 완료. 실제 체크포인트에서 훅이 layer_types대로 발화 |
| 2026-10-13 | Qwen3.5-0.8B (MPS, bf16) | run_grid 2x2 (KV 16/4 x S 16/4), WikiText-2 2 docs, prefill 128, decode 64, g128, seed 0 | mean NLL: (16,16)=3.389 ±0.381, (16,4)=4.430 ±0.330, (4,16)=3.433 ±0.415, (4,4)=4.457 ±0.320. 설정당 약 5s | M1 항목 2, 3 축소판 완료. fp 베이스라인 PPL 약 29.6. KV int4는 거의 무손실(+0.04), state int4 g128은 +1.04로 큰 손실. 2 docs 1 seed라 수치 자체는 결론 아님. state 포맷(group, symmetric, SR) 민감도가 다음 확인 대상 |
