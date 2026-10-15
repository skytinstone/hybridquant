# hybridquant

논문 **"Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs"**(Minseok Shin, 2026)의 코드, 실험 설정, 원본 결과를 담은 저장소입니다.

[English](README.md) | 한국어

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skytinstone/hybridquant/blob/main/notebooks/01_setup_colab.ipynb)

## 이 연구는 무엇인가

하이브리드 LLM은 추론하는 동안 두 가지 메모리를 들고 있습니다.

| 메모리 | 만드는 층 | 모양 | 문맥이 길어지면 커지는가 |
|---|---|---|---|
| KV 캐시 | 소프트맥스 어텐션 층 | `[B, H_kv, T, d]` | 커집니다 |
| 순환 상태(recurrent state) | Gated DeltaNet, Mamba-2 층 | `[B, H, d_k, d_v]` | 크기가 고정입니다 |

소프트맥스 어텐션은 지나간 토큰의 키와 값을 모두 저장해 두고 필요할 때 찾아 읽습니다. 그래서 메모리는 문맥 길이만큼 늘어나지만, 어떤 토큰이든 정확히 다시 꺼낼 수 있습니다. 선형 어텐션은 과거 전체를 크기가 고정된 행렬 하나에 압축해 담고, 토큰이 들어올 때마다 이 행렬을 조금씩 지우고 새로 씁니다. 메모리는 늘지 않지만, 한 번 생긴 반올림 오차가 그 자리에 머물지 않고 다음 단계로 계속 넘어갑니다. 두 메모리가 양자화에 전혀 다르게 반응하는 이유가 바로 이 차이입니다.

기존 연구는 두 메모리를 하나씩 따로 양자화했습니다. 이 연구는 질문을 바꿉니다. **전체 메모리 예산이 정해져 있을 때 비트를 두 메모리에 어떻게 나눠야 하는가, 그리고 두 오차는 서로 어떻게 얽히는가?**

모든 양자화는 PyTorch 안에서 양자화한 뒤 바로 되돌리는 **모의 양자화(fake quantization)**입니다. 전용 커널은 쓰지 않으며, 속도가 아니라 정확도와 메모리만 측정합니다.

## 주요 결과

Gated DeltaNet 하이브리드인 Qwen3.5 0.8B, 2B, 4B, 9B를 실험했고, Mamba-2 하이브리드인 Falcon-H1-1.5B를 대조군으로 썼습니다. 따로 적지 않은 수치는 모두 WikiText-2 문서 20개를 시드 3개로 측정한 값입니다.

- **두 메모리의 민감도 차이가 크고 일관됩니다.** KV 캐시는 어느 크기에서든 4비트로 줄여도 손실이 0.03 nats 이하입니다. 반면 상태를 4비트로 줄이면 0.3~1.8 nats를 잃고, bf16과 0.02 nats 안으로 맞추려면 8비트가 필요합니다.
- **원인은 델타 규칙일 가능성이 큽니다.** Falcon-H1에서는 Mamba-2 상태를 4비트로 줄여도 손실이 측정되지 않았습니다(+0.004 nats, 유의하지 않음). 다만 두 모델은 갱신 규칙 말고도 다른 점이 많아서, 이 비교는 델타 규칙 원인설을 뒷받침할 뿐 단정하지는 못합니다.
- **두 오차는 거의 그대로 더해집니다.** KV와 상태를 함께 4비트로 줄였을 때 손실은 각각의 손실을 더한 값보다 작은 모델에서 0.03~0.05 nats 더 클 뿐이고, 9B에서는 차이가 없습니다. 따라서 두 메모리의 비트는 거의 독립적으로 정해도 됩니다.
- **반올림은 오차를 매번 새로 만들지 않습니다.** 이미 반올림된 상태는 조금 바뀐 뒤 다시 반올림해도 같은 값으로 돌아오는 경우가 많아서, 토큰마다 새 오차가 생기지 않습니다. 그래서 반올림 대신 같은 크기의 가우시안 노이즈를 넣으면 손실이 9배로 커지고, 매번 무작위 오차를 넣는 확률적 반올림도 일반 반올림보다 5배 나쁩니다.
- **첫 번째 층이 거의 전부입니다.** 층 하나씩만 4비트로 줄여 보면, 첫 번째 GDN 층이 전체 민감도 합의 38%(0.8B), 80%(4B)를 차지합니다. 이 순서는 층의 감쇠 반감기와 관계가 없습니다.
- **어느 메모리가 발목을 잡는지는 과제마다 다릅니다.** 긴 문맥에서 정보를 찾는 NIAH는 KV 캐시에 좌우됩니다. 수학 추론(GSM8K, thinking 모드)은 상태에 좌우되어, NLL로는 손실이 없던 8비트 상태가 Qwen3.5-4B의 정답률을 0.600에서 0.345로 떨어뜨립니다.
- **오차 피드백(error feedback)이 상태 문제를 해결합니다.** 8비트 상태의 GSM8K 정답률을 bf16과 같은 0.600으로 되돌리고, 1,024토큰을 디코딩하는 동안 상태 오차가 커지지 않게 막습니다.

Qwen3.5-4B 권장 설정입니다. 메모리는 512토큰 기준 bf16 대비 비율입니다.

| 설정 | 메모리 | ΔNLL (nats) | GSM8K | NIAH |
|---|---|---|---|---|
| bf16 | 100% | 0 | 0.600 | 1.00 |
| KV 4비트 + 상태 6비트, 오차 피드백 | 32% | +0.044 | 0.555 (유의차 없음) | 1.00 |
| KV 4비트 + 상태 8비트, 오차 피드백 | 40% | −0.001 | 측정 안 함 | 1.00 |

전체 표는 `paper/tables/`에, 단계별 보고서는 `results/reports/`에 있습니다.

## 측정 방법

- **디코딩 시점 NLL.** 앞부분은 한 번에 prefill하고, 나머지는 양자화된 캐시를 쓰면서 한 토큰씩 넣어 teacher-forced NLL을 잽니다. 상태 오차는 이 방식에서만 쌓이기 때문에 이것을 주 지표로 씁니다. 문서 전체를 한 번에 통과시키는 일반적인 perplexity로는 상태의 영향이 보이지 않습니다.
- **일반 반올림과 오차 피드백 비교.** 상태를 양자화하는 모든 설정을 일반 반올림(RTN)과 오차 피드백(EF) 두 가지로 돌립니다. 오차 피드백은 저장할 때 생긴 반올림 잔차를 다음 저장 때 더해 주는 방식입니다.
- **노이즈 대조 실험.** 두 오차가 더해지는지 볼 때, 한쪽 양자화를 오차 크기(RMS)가 같은 가우시안 노이즈로 바꿔 봅니다. 오차의 크기와 양자화 고유의 구조를 떼어 보기 위해서입니다.
- **통계.** 설정마다 문서 20개를 시드 3개로 측정하며, 시드마다 서로 겹치지 않는 문서를 고릅니다. bf16과의 차이는 같은 문서끼리 짝지은 부트스트랩 95% 신뢰구간(1,000회)으로 보고합니다. GSM8K의 차이는 같은 문제끼리 짝지은 McNemar 정확 검정으로 판단합니다.
- **배치로 돌려도 결과는 같습니다.** `tests/test_quant_ext.py`와 `tests/test_falcon_batch.py`가 배치 디코딩과 문서 하나씩 디코딩한 NLL이 같은지 확인합니다.

## 폴더 구조

```
src/hybridquant/
  quant.py     모의 양자화 함수: 비대칭/대칭, 그룹, 확률적 반올림, 노이즈 대조, 오차 피드백
  hooks.py     transformers 캐시에 거는 저장 시점 훅(KV 저장, 상태 저장, 오차 피드백 잔차)과 훅 적용 확인
  compat.py    Falcon-H1용 transformers 우회 패치 2개(아래 "Falcon-H1" 참고)
  eval.py      배치 teacher-forced 디코딩 NLL, 배치 생성, 정답 채점, dtype 선택, 모델 로드
  data.py      WikiText-2와 C4를 이어 붙인 뒤 시드마다 겹치지 않는 구간으로 자르는 로더
  config.py    YAML 형식, 명령줄 덮어쓰기, 그리드 또는 개별 셀, 실행 목록
  budget.py    모델 설정에서 캐시 크기와 비트 예산 계산
  meta.py      실행 정보(git 커밋, 라이브러리 버전, GPU, 적용한 패치)를 결과 행마다 기록
configs/       단계마다 YAML 하나: 1라운드(smoke, grid, superadd, grid_4b_core), 2라운드(r2_*), 3라운드(r3_*)
scripts/
  run_grid.py       YAML 설정대로 디코딩 NLL을 측정합니다. 중단 후 이어서 돌릴 수 있습니다
  layer_sweep.py    GDN 층 하나씩만 상태를 양자화하고, 층별 감쇠 통계를 냅니다
  run_gsm8k.py      양자화 캐시로 GSM8K를 thinking 모드로 풉니다
  run_niah.py       4k, 16k 문맥에서 바늘 하나 찾기(NIAH)
  run_round1.sh, run_round2.sh, run_round3.sh, run_all.sh   단계마다 보고서 작성, 커밋, push까지 하는 실행기
  finish_run.py     실행이 끝나기를 기다렸다가 중복 제거, 보고서 재생성, Drive 복사, 커밋과 push
  dedupe_results.py results/*.jsonl의 중복 행 제거
  check_hooks.py    실제 체크포인트에서 훅이 호출되는지 PASS/FAIL로 확인
  diag_decode.py    캐시를 쓴 디코딩이 일반 forward와 같은 값을 내는지 확인
  vram_check.py, estimate_time.py, strip_tex_comments.py, 맥 동기화 스크립트
analysis/
  common.py, report.py, figures.py          결과 파일만으로 단계별 보고서와 그림 생성
  make_tex_tables.py                         paper/tables/의 LaTeX 표(Table 1, 부록 C)
  fig_overview.py, fig_memory.py, model_table.py   논문 그림 1, 그림 2, 모델 표
notebooks/     01 Colab 시작점, 02 수식 검증, 03 결과 대시보드
tests/         테스트 64개, CPU만 사용, 체크포인트 다운로드 없음
docs/          experiment_log.md, decision_log.md, related_work.md, refs_verification.md
paper/         figures/, tables/, refs_verified.bib
results/       jsonl 결과, reports/, figures/, 라운드 로그(git으로 관리), 제외한 실행은 results/invalid/
```

## 훅의 동작 방식

`transformers`는 두 메모리를 모두 `DynamicCache`에 담습니다. 캐시의 층은 어텐션용 `DynamicLayer`, GDN과 Mamba-2용 `LinearAttentionLayer`, 그리고 둘을 함께 가진 Falcon-H1용 `LinearAttentionAndFullAttentionLayer`가 있습니다. 여기서 두 메서드만 바꿉니다.

* `DynamicLayer.update(k, v)`: 새 토큰의 K, V를 캐시에 붙이기 전에 토큰 단위로 양자화합니다.
* `LinearAttentionLayer.update_recurrent_state(S)`: 상태 `S`를 저장하기 전에 양자화합니다.

그래서 다음 디코딩 단계는 반올림된 값을 읽게 됩니다. 모델 코드는 전혀 수정하지 않습니다.

```python
from hybridquant import HybridQuantConfig, load_model, decode_nll
model, tok = load_model("Qwen/Qwen3.5-0.8B")
cfg = HybridQuantConfig.uniform(kv_bits=4, state_bits=8, state_error_feedback=True)
ids = tok(text, return_tensors="pt")["input_ids"]
r = decode_nll(model, ids, prefill_len=256, cfg=cfg, max_decode=256)
print(r.ppl, r.stats)
```

prefill 동안에는 청크 커널이 중간 상태를 full precision으로 유지하고, 마지막 상태만 한 번 양자화합니다. 따라서 상태 오차는 한 토큰씩 디코딩할 때만 쌓이며, `decode_nll`이 이어지는 부분을 한 토큰씩 넣는 것도 이 때문입니다.

## Falcon-H1

이 저장소가 고정한 transformers 버전에는 Falcon-H1에 영향을 주는 문제가 두 가지 있고, `compat.py`가 둘 다 우회합니다. `load_model`이 자동으로 적용하고, 어떤 패치를 적용했는지 출력한 뒤 모든 결과 행에 기록합니다.

1. 한 토큰씩 디코딩할 때 `FalconH1RMSNormGated`가 `[B, dim]` 입력과 `[B, 1, dim]` 게이트를 브로드캐스트하면서, 배치 안의 서로 다른 문서가 섞입니다.
2. 한 토큰씩 디코딩할 때 Mamba-2 상태가 제자리에서 갱신되어 `update_recurrent_state`를 거치지 않습니다. 그래서 상태 훅이 이를 보지 못합니다. 패치는 토큰마다 상태를 캐시를 통해 다시 저장하게 합니다.

이 수정 전에 나온 Falcon-H1 결과는 `results/invalid/`에 따로 보관하며 분석에 쓰지 않습니다. 자세한 경위는 `docs/decision_log.md`에 있습니다.

## 훅 확인

GPU에서는 `kernels` 허브가 `Qwen3_5GatedDeltaNet.forward`를 통째로 바꿔, 상태 훅이 아무 경고 없이 빠질 수 있습니다. 실험 전에 꼭 확인하십시오.

```bash
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
# state_calls=0으로 FAIL이 나오면:
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4 --no-hub-kernels
```

## 논문 실험 환경

논문의 모든 실험은 Colab Pro+의 NVIDIA A100-SXM4-40GB 한 장에서 bf16, PyTorch 2.11, transformers 5.19.0.dev0(커밋 `66e0f4a`, `requirements.txt`에 고정)로 돌렸습니다. GDN과 Mamba-2 순환 계산은 flash-linear-attention이나 causal-conv1d 없이 transformers의 순수 PyTorch 경로로 실행했습니다. 수치를 그대로 재현하려면 노트북의 GPU 선택 셀에서 고속 커널을 설치하지 마십시오.

`load_model(dtype="auto")`는 compute capability 8.0 이상과 Apple MPS에서 bf16, T4 같은 이전 세대 CUDA GPU에서 fp16, CPU에서 fp32를 고릅니다. `HQ_DTYPE` 환경변수나 `--dtype`으로 바꿀 수 있습니다.

## 논문 재현

세 라운드를 Colab에서 차례로 돌렸습니다. 실행기는 중간에 끊겨도 이어서 돌릴 수 있고, 단계가 끝날 때마다 `results/reports/<stage>.md`와 그림을 만든 뒤 `results/`를 커밋하고 push합니다.

```bash
bash scripts/run_all.sh        # 1, 2라운드: 0.8B 그리드, 가산성, 층별 민감도, 4B 핵심 셀, 형식 비교, C4, 2B, 9B, GSM8K, NIAH
bash scripts/run_round3.sh     # 3라운드: Falcon-H1 재실행, 4B 권장 설정과 그 NIAH, GSM8K 상태 8비트+EF,
                               #          4B 층별 민감도, 0.8B와 4B의 1,024토큰 디코딩
python scripts/finish_run.py   # 중복 제거, 모든 보고서 재생성, 커밋과 push
```

유효한 실행에 든 시간은 A100 기준 약 23시간입니다. 단계별 시간은 `paper/tables/compute.md`에 있습니다.

단계 하나만 돌릴 때는 다음과 같이 합니다.

```bash
python scripts/run_grid.py --config configs/grid_0.8b.yaml
python scripts/layer_sweep.py --model Qwen/Qwen3.5-0.8B --state-bits 4 --n-docs 8
python scripts/run_gsm8k.py --config configs/r2_gsm8k_4b.yaml
python scripts/run_niah.py --config configs/r2_niah_4b.yaml
```

보고서, 그림, 논문 표는 결과 파일만으로 다시 만들 수 있습니다.

```bash
python analysis/report.py --stage grid_0.8b --inputs results/grid_0.8b.jsonl --out results/reports/grid_0.8b.md
python analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md
python analysis/make_tex_tables.py      # paper/tables/*.tex
```

## Colab에서 돌리기

1. 위의 배지를 눌러 노트북을 열고 A100 런타임을 고릅니다.
2. **0. 토큰 설정**: Colab 보안 비밀(Secrets)에 `GH_TOKEN`(push용)과 `HF_TOKEN`(모델 다운로드용)을 넣습니다. 토큰이 없으면 결과는 Drive로만 복사됩니다.
3. **1. GPU 선택**: dtype과 고속 커널 설치 여부를 정합니다.
4. 훅 확인 셀이 PASS인지 본 뒤, 필요하면 Drive를 연결하고 백그라운드 실행 셀을 실행합니다. 연결이 끊기면 같은 셀을 다시 실행하면 되고, 이미 끝난 측정은 건너뜁니다.
5. 진행 상황은 모니터 셀이나 `notebooks/03_results_dashboard.ipynb`에서 봅니다.

## 맥에서 확인하기

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch                 # CPU만 쓸 때는 --index-url https://download.pytorch.org/whl/cpu 를 붙입니다
pip install -r requirements.txt pytest
pip install -e .
pytest tests -q
```

Apple Silicon에서는 `load_model`이 MPS를 자동으로 고르고, Qwen3.5-0.8B가 그대로 돌아갑니다.

## 진행 상황

완료했습니다. 세 라운드의 실험이 모두 끝났고, 논문의 모든 수치는 `results/`에 있습니다. 결과 기록은 [docs/experiment_log.md](docs/experiment_log.md), 설계 결정은 [docs/decision_log.md](docs/decision_log.md)에 있습니다.

## 데이터

- WikiText-2(raw): Merity et al., 2016. `Salesforce/wikitext`의 `wikitext-2-raw-v1`을 씁니다. test 분할은 NLL에, train 분할은 NIAH의 배경 글에 씁니다.
- C4(en): Raffel et al., 2020. `allenai/c4`의 validation 샤드 0에서 앞 5,000개 문서를 씁니다.
- GSM8K: Cobbe et al., 2021. `openai/gsm8k`의 `main` 설정 test 분할에서 고정한 200문제(`results/gsm8k_subset200.json`)를 씁니다.

## 인용

```bibtex
@misc{shin2026hybridquant,
  title  = {Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs},
  author = {Shin, Minseok},
  year   = {2026},
  url    = {https://github.com/skytinstone/hybridquant}
}
```

`CITATION.cff`도 함께 참고하십시오.

## 라이선스

MIT입니다. [LICENSE](LICENSE)를 참고하십시오.
