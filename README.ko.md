# hybridquant

하이브리드 선형 어텐션 LLM(Qwen3.5, Falcon-H1)의 KV cache와 recurrent state를 함께 양자화하는 연구 코드입니다.

[English](README.md) | 한국어

> English README.md is the source of truth; this is a Korean summary.
> 영문 README.md가 기준 문서입니다. 이 문서는 한국어 요약이며, 두 문서가 다르면 영문을 따릅니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skytinstone/hybridquant/blob/main/notebooks/01_setup_colab.ipynb)

## 무엇을 하는 코드인가

하이브리드 LLM은 추론 중에 두 종류의 메모리를 유지합니다. KV cache는 컨텍스트에 비례해 커지고 recurrent state는 고정 크기입니다. 일반(softmax) attention은 과거 토큰의 key와 value를 전부 보관하고 유사도로 찾아보므로 메모리가 컨텍스트에 비례해 커지지만 어떤 토큰이든 정확히 꺼낼 수 있고, 선형 attention(Gated DeltaNet, Mamba-2)은 과거를 고정 크기 행렬 하나에 접어 넣고 토큰마다 감쇠하며 고쳐 쓰므로 메모리는 일정하지만 한 번 들어간 반올림 오차가 다음 스텝으로 되먹임됩니다. 오차가 살아남는 방식의 이 차이가 두 메모리가 양자화에 전혀 다르게 반응하는 이유입니다. 선행 연구는 두 메모리를 각각 따로 양자화했습니다. 이 레포는 총 메모리 예산이 정해져 있을 때 비트를 둘 사이에 어떻게 나눠야 하고 두 오차가 서로 어떤 영향을 주는지를 묻습니다. 모든 계산은 fake quantization(양자화 후 바로 역양자화)이고 커스텀 커널은 없으며, 정확도와 메모리만 측정합니다.

## 무엇을 측정하는가

- **decode 시점 NLL.** prefill 뒤 continuation을 한 토큰씩 넣는 teacher-forced NLL입니다. state 오차는 이 경로에서만 쌓이므로 1차 지표이며, 단순 forward PPL로는 state 효과가 보이지 않습니다.
- **RTN 대 error feedback.** state를 양자화하는 모든 셀을 RTN과 EF(저장 잔차를 다음 저장에 더함) 두 번 돌리고, EF를 모든 표의 비교축으로 둡니다.
- **노이즈 대조 조건.** 초가산 검증에서 한쪽의 양자화를 같은 오차 RMS의 가우시안 노이즈로 바꿔 오차 크기와 양자화 구조를 분리합니다.
- **통계.** 셀당 문서 20개 × 시드 3개입니다. 시드는 겹치지 않는 문서 블록을 고르므로 시드 분산은 문서 샘플 분산입니다. fp 대비 차이는 문서 대응 부트스트랩 95% CI(1000회)로 보고하고, CI가 0을 포함하면 "구분 불가"로 표기합니다.
- **배치 decode는 결과를 바꾸지 않습니다.** 길이가 같은 문서를 패딩 없이 배치로 돌리며, 배치와 단일 문서의 NLL이 같은지 테스트로 확인합니다.

## 구조

영문 README의 Layout과 같습니다. 핵심만 적으면 src/hybridquant/는 quant(양자화), hooks(캐시 훅과 EF), eval(배치 decode와 생성), data(WikiText-2와 C4 창 로더), config(YAML과 셀 목록), budget(식 1과 5), meta(실행 메타데이터)이고, configs/는 라운드 1과 2의 YAML 13개, scripts/는 실행기(run_grid, layer_sweep, run_gsm8k, run_niah)와 러너(run_round1, run_round2, run_all), 점검 도구(check_hooks, vram_check, estimate_time), 맥 동기화 스크립트입니다. analysis/는 보고서와 그림 생성, notebooks/는 Colab 진입점(01), 수식 검증(02), 결과 대시보드(03)입니다. results/는 git으로 추적합니다.

## 훅 동작 방식

transformers의 DynamicCache 레이어 두 메서드를 패치합니다. `DynamicLayer.update`는 새 토큰의 K/V를 저장 전에, `LinearAttentionLayer.update_recurrent_state`는 state를 저장 전에 양자화합니다. Falcon-H1의 hybrid 레이어는 두 클래스를 모두 상속하므로 같은 훅이 걸립니다. 모델 코드는 수정하지 않습니다.

결과를 읽을 때 한 가지를 염두에 두어야 합니다. prefill 중에는 중간 state가 full precision이고 최종 state만 양자화되므로 state 오차는 토큰 단위 decode 중에만 쌓입니다.

## 훅 검증

GPU에서는 kernels 허브가 GatedDeltaNet forward를 통째로 바꿔 state 훅을 건너뛸 수 있습니다. 실험 전에 `scripts/check_hooks.py`로 PASS를 확인하고, FAIL이면 `--no-hub-kernels`로 다시 실행합니다.

## GPU와 dtype

`load_model(dtype="auto")`는 compute capability 8.0 이상(A100, H100, L4)과 MPS에서 bf16, T4 같은 구형 CUDA에서 fp16, CPU에서 fp32를 고릅니다. 즉 T4에서는 실험이 fp16으로 돕니다. GDN 점화식과 decay 계산, fake 양자화는 fp32로 수행되고, decode_nll은 비유한 NLL이 나오면 fp32 재실행 힌트와 함께 중단합니다. 수식 검증 노트북 02만 T4에서 fp32를 씁니다. T4에서는 고속 커널을 설치하지 않고 순수 torch 폴백을 씁니다.

## Colab 워크플로

1. 배지를 열거나 Colab에서 레포 URL로 노트북을 엽니다. T4, L4, A100, H100 중 하나를 고릅니다.
2. **0. 토큰 설정** 셀에서 GH_TOKEN(비공개 레포, push)과 HF_TOKEN을 Colab Secrets로 넣습니다. 토큰이 없으면 셀은 경고만 하고 넘어가지만, 그 경우 결과는 push되지 않고 Drive로만 복사됩니다.
3. **1. GPU 선택** 셀이 카드를 감지해 dtype과 커널 설치 여부를 정합니다.
4. 훅 검증 셀이 PASS인지 본 뒤, 필요하면 Drive를 마운트하고 **백그라운드 실행** 셀을 실행합니다. run_all.sh가 라운드 1과 2를 순서대로 돌리며 단계마다 보고서와 그림을 만들고 results/를 커밋해 push합니다. 세션이 끊기면 같은 셀을 다시 실행하면 끝난 행은 건너뜁니다.
5. 진행은 진행 확인 셀이나 notebooks/03_results_dashboard.ipynb로 봅니다. 맥에서는 `scripts/install_sync_launchd.sh`가 30분마다 pull해서 ~/Downloads/hybridquant_results/날짜/에 복사합니다.

노트북 셀을 손으로 하나씩 실행하는 것은 디버깅용입니다.

## 논문 재현

```bash
bash scripts/run_round1.sh      # smoke_0.8b, grid_0.8b, superadd_0.8b, layer_sweep(0.8B), grid_4b_core
bash scripts/run_round2.sh      # ablation, c4, niah 0.8B, 2B, gsm8k 4B, niah 4B, 9B, Falcon-H1 1.5B
bash scripts/run_all.sh         # 두 라운드와 종합 보고서
python analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md
python analysis/figures.py --stage grid_0.8b --inputs results/grid_0.8b.jsonl
```

라운드 3(외부 H100의 27B)은 아직 정의돼 있지 않아 명령이 없습니다. 표는 results/reports/<stage>.md, 그림은 results/figures/<stage>/에 PDF와 PNG로 저장됩니다. 노트북 02의 수식 그림은 paper/figures/에 저장되고, 논문 소스로 옮기는 일은 아직 수동입니다(paper/tables/는 없습니다).

## 로컬 검증 (맥)

CPU만 쓸 때는 `pip install torch --index-url https://download.pytorch.org/whl/cpu`, Apple Silicon에서 MPS를 쓸 때는 index-url 없이 `pip install torch`를 설치한 뒤 `pip install -r requirements.txt pytest`와 `pip install -e .`를 하고 `pytest tests -q`로 확인합니다. MPS에서는 Qwen3.5-0.8B가 그대로 돌아갑니다. requirements.txt는 transformers를 라운드 1/2에서 쓴 커밋(66e0f4a)으로 고정하며 라운드 3 시점에 다시 고정합니다.

## 진행 상황

현재 마일스톤은 **M2**(비트 배분 그리드, EF, 초가산, 레이어 스윕, 라운드 1 Colab 실행 중)입니다. 결과는 [docs/experiment_log.md](docs/experiment_log.md), 설계 결정은 [docs/decision_log.md](docs/decision_log.md)에 있습니다.

## 데이터, 인용, 라이선스

WikiText-2(Merity et al. 2016, Salesforce/wikitext), C4(Raffel et al. 2020, allenai/c4 validation 샤드 0의 앞 5000문서), GSM8K(Cobbe et al. 2021, openai/gsm8k test 200문제 고정 부분집합)를 씁니다. 인용 정보는 영문 README의 Citation과 CITATION.cff에 있으며 arXiv 번호는 아직 없습니다. 라이선스는 MIT입니다.
