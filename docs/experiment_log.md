# 실험 로그

형식은 날짜 | 모델 | 설정 | 결과 | 결론이며, 한 실험을 한 줄로 적습니다. 결과 파일은 results/ 에 jsonl로 보관합니다.

| 날짜 | 모델 | 설정 | 결과 | 결론 |
|---|---|---|---|---|
| 2026-10-13 | (없음) | 레포, 훅, 평가 코드 뼈대 작성 | CPU 스모크 테스트 결과는 아래 줄에 있습니다 | M1을 시작했습니다 |
| 2026-10-13 | tiny Qwen3.5 (random init, hidden 64, 4 layers, CPU) | tests/test_smoke_cpu.py, transformers 5.19.0.dev0 | CPU 스모크 테스트 12개가 통과했습니다. 훅 발화, 레이어 인덱스 일치, (16,16)과 (2,2)의 decode NLL 차이, 패치 원복을 확인했습니다 | M1 로컬 검증을 마쳤습니다. 다음은 실제 체크포인트 Qwen3.5-0.8B입니다 |
| 2026-10-13 | Qwen3.5-0.8B (MPS, M3 Pro, bf16) | scripts/check_hooks.py, KV2/S4, prefill 128, decode 32 | hook coverage PASS입니다. layers_seen_kv=[3,7,11,15,19,23], layers_seen_state는 나머지 18개, use_kernels=False, 모델 로드 포함 70초가 걸렸습니다 | M1 항목 1을 마쳤습니다. 실제 체크포인트에서 훅이 layer_types대로 발화합니다 |
| 2026-10-13 | Qwen3.5-0.8B (MPS, bf16) | run_grid 2x2 (KV 16/4 x S 16/4), WikiText-2 2 docs, prefill 128, decode 64, g128, seed 0 | mean NLL은 (16,16)=3.389 ±0.381, (16,4)=4.430 ±0.330, (4,16)=3.433 ±0.415, (4,4)=4.457 ±0.320이고 설정당 약 5초 걸렸습니다 | M1 항목 2, 3의 축소판을 마쳤습니다. fp 베이스라인 PPL은 약 29.6입니다. KV int4는 거의 손실이 없고(+0.04) state int4 g128은 손실이 큽니다(+1.04). 2 docs 1 seed라 수치 자체로 결론을 내리지는 않습니다. state 포맷(group, symmetric, SR) 민감도를 다음에 확인합니다 |
| 2026-10-13 | Qwen3.5-0.8B (Colab A100, bf16) | 노트북 01 오차-위치 곡선, (16,16) (16,4) (4,16) (4,4), prefill 256, decode 256. 1문서/1시드, 같은 문장 40회 반복 텍스트, 예비 관찰 | (a) state 4bit는 fp 대비 NLL이 10배 이상 올랐고 KV 4bit는 fp와 거의 같았습니다. (b) state 오차는 decode 시작 직후 최고점을 찍고 50토큰 안에 감쇠했습니다. (c) KV4+S4가 S4 단독보다 낮게 나오는 역전이 1개 문서에서 보였습니다(노이즈 가능성, 미확정). 반복 텍스트라 절대값은 의미가 없습니다 | WikiText 그리드로 재확인 필요 |
