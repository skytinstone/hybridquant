# hybridquant: 프로젝트 컨텍스트 (Claude Code 로컬 세션용)

## 연구 한 줄 요약
하이브리드 선형 어텐션 LLM(Qwen3.5: Gated DeltaNet + Gated Attention)에서 KV cache와 recurrent state를 **공동으로** fake-quantize하여, 고정 메모리 예산에서의 비트 배분과 두 오차의 교차 전파를 분석한다. 단독 저자 SCI 논문. 가제 "Where Do the Bits Go?".

## 제약
- Google Colab Pro(L4/A100)에서 실행. 커스텀 CUDA 커널 금지. PyTorch fake-quant만. 속도 측정 제외.
- 모델: Qwen/Qwen3.5-0.8B → 2B → 4B → 9B. 2차 아키텍처 Falcon-H1 0.5B~3B(Mamba-2 하이브리드).
- 비교 대상: DAMP, LeapQuant, STEPQuant (2026.08~09, 전부 state 단독 양자화).

## 코드 구조와 핵심 설계
- `src/hybridquant/quant.py` fake-quant 프리미티브. 마지막 차원 기준 group quant. bits>=16이면 identity.
- `src/hybridquant/hooks.py` transformers 캐시 클래스의 쓰기 메서드를 패치:
  - `DynamicLayer.update(k, v)` → 새 토큰 K/V를 per-token group으로 양자화 후 저장
  - `LinearAttentionLayer.update_recurrent_state(S)` → state [B,H,dk,dv]를 양자화 후 저장
  - 모델 코드는 건드리지 않는다. `quantized_cache(cfg)` 컨텍스트 매니저로 on/off.
- `src/hybridquant/eval.py`
  - `decode_nll`: prefill 후 continuation을 **한 토큰씩** 넣는 teacher-forced NLL. state 오차는 decode에서만 누적되므로 이것이 1차 지표. 단순 forward PPL은 state 효과를 못 본다.
  - `generate_and_score`: thinking 모드 생성 + 행동 통계(토큰 수, 예산 소진, 8-gram 반복률).
- `scripts/run_grid.py` KV bits × state bits 그리드 → `results/*.jsonl`
- `notebooks/01_setup_colab.ipynb` Colab 진입점.

## 작업 규칙
- 실험 결과는 `docs/experiment_log.md`에 한 줄씩 누적. 설계 변경은 `docs/decision_log.md`에 추가(번복해도 이전 줄 삭제 금지). 새 논문은 `docs/related_work.md`.
- 결과 해석 시 시드/샘플 수/분산 먼저 확인. 한 번의 수치로 결론 내리지 않는다.
- 코드 변경은 작게, 지정된 부분만. em-dash 사용 금지. 논의는 한국어, 논문 문장은 영어.
- novelty 판단은 반드시 최신 논문 검색 후.

## 현재 마일스톤: M1 (환경 구축)
1. Colab에서 Qwen3.5-0.8B 로드, 훅이 실제로 호출되는지 `QuantStats`로 확인
2. fp16 베이스라인 decode NLL 재현
3. 0.8B에서 4×3 그리드 1회 실행 → 파이프라인 검증
4. Falcon-H1-0.5B에서 같은 훅이 동작하는지 확인
