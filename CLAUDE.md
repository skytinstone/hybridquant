# hybridquant: 프로젝트 컨텍스트 (Claude Code 로컬 세션용)

## 연구 한 줄 요약
하이브리드 선형 어텐션 LLM(Qwen3.5: Gated DeltaNet + Gated Attention)에서 KV cache와 recurrent state를 **함께** fake-quantize하고, 고정된 메모리 예산 안에서 비트를 어떻게 나눌지와 두 오차가 서로 어떻게 번지는지를 분석합니다. 단독 저자 SCI 논문이며 가제는 "Where Do the Bits Go?"입니다.

## 제약
- 실행 환경은 Google Colab Pro(T4/L4/A100/H100)와 로컬 맥(M3 Pro, torch MPS)입니다. 커스텀 CUDA 커널은 쓰지 않고 PyTorch fake-quant만 사용합니다. 속도는 측정하지 않습니다.
- 모델은 Qwen/Qwen3.5-0.8B → 2B → 4B → 9B 순서로 키웁니다. 2차 아키텍처로 Falcon-H1 0.5B~3B(Mamba-2 하이브리드)를 씁니다.
- 비교 대상은 DAMP, LeapQuant, STEPQuant입니다(2026.08~09, 모두 state만 양자화).

## 코드 구조와 핵심 설계
- `src/hybridquant/quant.py`: fake-quant 프리미티브입니다. 마지막 차원을 기준으로 group 양자화하며, bits가 16 이상이면 identity입니다.
- `src/hybridquant/hooks.py`: transformers 캐시 클래스의 쓰기 메서드를 패치합니다.
  - `DynamicLayer.update(k, v)`: 새 토큰의 K/V를 per-token group으로 양자화한 뒤 저장합니다.
  - `LinearAttentionLayer.update_recurrent_state(S)`: state [B,H,dk,dv]를 양자화한 뒤 저장합니다.
  - 모델 코드는 건드리지 않습니다. `quantized_cache(cfg)` 컨텍스트 매니저로 켜고 끕니다.
- `src/hybridquant/eval.py`
  - `decode_nll`: prefill 후 continuation을 **한 토큰씩** 넣는 teacher-forced NLL입니다. state 오차는 decode에서만 쌓이므로 이것이 1차 지표입니다. 단순 forward PPL로는 state 효과가 보이지 않습니다.
  - `generate_and_score`: thinking 모드로 생성하고 행동 통계(토큰 수, 예산 소진 여부, 8-gram 반복률)를 냅니다.
- `scripts/check_hooks.py`: 실제 체크포인트에서 훅이 layer_types대로 발화하는지 PASS/FAIL로 판정합니다.
- `scripts/run_grid.py`: KV bits × state bits 그리드를 돌려 `results/*.jsonl`에 씁니다.
- `notebooks/01_setup_colab.ipynb`: Colab 진입점입니다.

## 작업 규칙
- 실험 결과는 `docs/experiment_log.md`에 한 줄씩 쌓습니다. 설계 변경은 `docs/decision_log.md`에 추가하고, 번복하더라도 이전 줄은 지우지 않습니다. 새로 찾은 논문은 `docs/related_work.md`에 적습니다.
- 결과를 해석할 때는 시드, 샘플 수, 분산을 먼저 확인합니다. 한 번 나온 수치로 결론을 내리지 않습니다.
- 코드 변경은 작게, 지정된 부분만 합니다. em-dash는 쓰지 않습니다. 논의는 한국어로, 논문 문장은 영어로 씁니다. 한국어 문장은 "습니다"체로 씁니다.
- novelty 판단은 반드시 최신 논문을 검색한 뒤에 합니다.

## 현재 마일스톤: M1 (환경 구축)
1. Qwen3.5-0.8B를 로드하고 훅이 실제로 호출되는지 `QuantStats`로 확인합니다.
2. fp 베이스라인 decode NLL을 재현합니다.
3. 0.8B에서 4×3 그리드를 1회 돌려 파이프라인을 검증합니다.
4. Falcon-H1-0.5B에서 같은 훅이 동작하는지 확인합니다.
