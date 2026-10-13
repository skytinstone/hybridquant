# 실험 로그

형식: 날짜 | 모델 | 설정 | 결과 | 결론. 한 실험 한 줄. 결과 파일은 results/ 에 jsonl로 보관.

| 날짜 | 모델 | 설정 | 결과 | 결론 |
|---|---|---|---|---|
| 2026-10-13 | (없음) | 레포/훅/평가 코드 뼈대 작성 | CPU 스모크 테스트 통과 여부는 아래 기록 | M1 시작 |
| 2026-10-13 | tiny Qwen3.5 (random init, hidden 64, 4 layers, CPU) | tests/test_smoke_cpu.py, transformers 5.19.0.dev0 | CPU 스모크 테스트 통과 (12 passed): 훅 발화, 레이어 인덱스 일치, (16,16) vs (2,2) decode NLL 상이, 패치 원복 | M1 로컬 검증 완료, 다음은 Colab에서 Qwen3.5-0.8B |
