# hybridquant

하이브리드 선형 어텐션 LLM(Qwen3.5, Falcon-H1)의 KV cache와 recurrent state를 함께 양자화하는 연구 코드입니다.

[English](README.md) | 한국어

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skytinstone/hybridquant/blob/main/notebooks/01_setup_colab.ipynb)

## 무엇을 하는 코드인가

하이브리드 LLM은 추론 중에 두 종류의 메모리를 유지합니다.

| 메모리 | 생성 레이어 | 형태 | 컨텍스트에 따라 커지는지 |
|---|---|---|---|
| KV cache | softmax 어텐션 레이어 | `[B, H_kv, T, d]` | 커집니다 |
| recurrent state | Gated DeltaNet / Mamba-2 레이어 | `[B, H, d_k, d_v]` | 고정 크기입니다 |

선행 연구는 둘 중 하나만 따로 양자화했습니다. 이 레포는 다음 질문을 다룹니다. **총 메모리 예산이 정해져 있을 때 비트를 둘 사이에 어떻게 나눠야 하고, 두 오차는 서로 어떤 영향을 주는가?**

모든 계산은 **fake quantization**(PyTorch에서 양자화한 뒤 바로 역양자화)입니다. 커스텀 커널은 없습니다. 정확도와 메모리를 측정하고, 지연 시간은 측정하지 않습니다.

## 구조

```
src/hybridquant/
  quant.py    fake-quant 프리미티브 (asym/sym, group, stochastic rounding)
  hooks.py    transformers 캐시 레이어의 저장 시점 훅 (KV 쓰기, state 쓰기) + 훅 커버리지 판정
  eval.py     decode 시점 NLL (teacher-forced), 생성 + 행동 통계, dtype 자동 결정
scripts/
  check_hooks.py  실제 체크포인트에서 훅이 layer_types대로 발화하는지 PASS/FAIL 판정
  run_grid.py     WikiText-2에서 비트 배분 그리드 -> results/*.jsonl
notebooks/
  01_setup_colab.ipynb  Colab 설정 + 스모크 테스트
tests/
  test_smoke_cpu.py     작은 랜덤 초기화 Qwen3.5로 CPU에서 훅과 양자화 검증
docs/
  experiment_log.md decision_log.md related_work.md
```

## 훅 동작 방식

`transformers`는 두 메모리를 모두 `DynamicCache`에 저장하고, 각 레이어는 `DynamicLayer`(어텐션) 또는 `LinearAttentionLayer`(GDN/Mamba)입니다. 다음 두 메서드를 패치합니다.

* `DynamicLayer.update(k, v)`: 새 토큰의 K/V를 append 전에 per-token으로 양자화합니다.
* `LinearAttentionLayer.update_recurrent_state(S)`: `S`를 저장 전에 양자화합니다.

그래서 다음 decode 스텝은 반올림된 state를 읽게 됩니다. 모델 코드는 수정하지 않습니다.

```python
from hybridquant import HybridQuantConfig, load_model, decode_nll
model, tok = load_model("Qwen/Qwen3.5-0.8B")
cfg = HybridQuantConfig.uniform(kv_bits=4, state_bits=8)
ids = tok(text, return_tensors="pt")["input_ids"]
r = decode_nll(model, ids, prefill_len=256, cfg=cfg, max_decode=256)
print(r.ppl, r.stats)
```

한 가지 짚고 갈 점이 있습니다. prefill 중에는 chunked 커널이 중간 state를 full precision으로 유지하고 최종 state만 양자화됩니다. 따라서 state 오차는 토큰 단위 decode 중에만 쌓이고, 그래서 `decode_nll`이 continuation을 한 토큰씩 넣습니다.

## 훅 검증

GPU에서는 `kernels` 허브가 `Qwen3_5GatedDeltaNet.forward`를 통째로 바꿔치기해 state 훅을 조용히 건너뛸 수 있습니다. 실험 전에 반드시 확인합니다.

```bash
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
# FAIL이고 state_calls=0이면:
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4 --no-hub-kernels
```

`--no-hub-kernels`는 transformers를 import하기 전에 `USE_HUB_KERNELS=NO`를 설정하고 `from_pretrained(use_kernels=False)`를 넘깁니다. 근거는 `docs/decision_log.md`에 있습니다.

## Colab 워크플로

1. 위 배지를 열거나, Colab에서 `File > Open notebook > GitHub`에 레포 URL을 붙여넣습니다.
2. **0. 토큰 설정** 셀에서 토큰을 넣습니다. 비공개 레포라면 `GH_TOKEN`, HF 모델 다운로드에는 `HF_TOKEN`이 필요합니다. Colab Secrets(왼쪽 사이드바 열쇠 아이콘)에 같은 이름으로 저장해 두면 셀이 자동으로 읽고, 없으면 경고만 하고 넘어갑니다.
3. **1. GPU 선택** 셀이 연결된 카드를 감지해 dtype(T4는 fp16, 나머지는 bf16)과 고속 커널 설치 여부(T4는 건너뛰고 순수 torch 폴백)를 정합니다. T4, L4, A100, H100을 지원합니다.
4. 결과는 `results/*.jsonl`에 쌓입니다. 마지막 셀로 레포에 커밋하거나 Drive로 복사합니다.
5. Colab에서 노트북을 고쳐 저장하려면 `File > Save a copy in GitHub`를 씁니다.

## 로컬 검증 (맥, CPU 또는 MPS)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" accelerate datasets pytest
pip install -e .
pytest tests -q
```

Apple Silicon이면 0.8B는 MPS에서 바로 돌아갑니다. `load_model`의 기본값이 MPS를 자동으로 잡습니다.

## 진행 상황

`docs/experiment_log.md`를 참고하십시오. 현재 마일스톤은 **M1**(환경 + 훅 + 베이스라인)입니다.
