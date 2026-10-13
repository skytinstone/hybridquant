# 결정 로그

번복 시 이전 결정을 지우지 않고 아래에 추가한다.

- 2026-10-13 연구 주제 확정: 하이브리드 LLM의 KV cache + recurrent state 공동 양자화(비트 배분). 이유: 선형 어텐션 state 양자화 논문이 2026.08~09에 4편 나왔지만 전부 state 단독, joint allocation 없음. dLLM/VLA/MLA 양자화는 이미 선점됨.
- 2026-10-13 주력 모델 Qwen3.5 0.8B/2B/4B/9B. 2차 아키텍처로 Falcon-H1(Mamba-2 하이브리드) 0.5B~3B 추가 예정. 이유: GDN(delta rule)과 Mamba-2(decay only)에서 같은 경향이 나와야 일반화 주장 가능.
- 2026-10-13 양자화 개입 지점: transformers 캐시 클래스의 쓰기 메서드(DynamicLayer.update, LinearAttentionLayer.update_recurrent_state). 이유: 모델 코드 무수정, 커널 무관, 모든 하이브리드 모델에 동일 적용.
- 2026-10-13 state 양자화 의미론: "저장 시점 양자화". prefill 중간 state는 full precision, 최종 state만 양자화. decode 중에는 매 토큰 양자화. 따라서 1차 지표는 teacher-forced decode NLL. 단순 forward PPL은 state 효과를 못 본다.
- 2026-10-13 v1 양자화 포맷: K/V는 per-token asymmetric group(128) INT. state는 d_v 축 group(128) asymmetric INT, stochastic rounding 옵션. KIVI식 per-channel key는 v2에서.
- 2026-10-13 속도 측정 제외, fake-quant만. 이유: Colab 단독, 커스텀 커널 금지.
- 2026-10-13 hooks.py: state 훅의 state_calls / layers_seen_state 카운터를 양자화 비활성(bits>=16)일 때도 증가시키도록 변경(KV 훅과 동일). 이유: fp16 베이스라인에서 QuantStats로 훅 발화 여부를 확인해야 하는데(M1 항목 1), 기존 코드는 state_calls=0을 보고해 미발화처럼 보였다. 양자화 자체는 여전히 enabled일 때만 적용.
- 2026-10-13 M1 로컬 검증: transformers 5.19.0.dev0(git main) 기준 DynamicLayer.update(k, v, *args, **kwargs)와 LinearAttentionLayer.update_recurrent_state(s, state_idx=0, **kwargs) 시그니처가 패치 함수와 일치. Qwen3_5GatedDeltaNet.forward와 Qwen3_5Attention.forward는 각각 Cache.update_recurrent_state / Cache.update를 거쳐 레이어 클래스 메서드를 호출하므로 클래스 패치가 항상 경로에 있다. DynamicCache(config)는 layer_types 순서대로 LinearAttentionLayer / DynamicLayer를 생성하므로 tag_cache_layers의 enumerate 인덱스가 decoder layer_idx와 일치.
- 2026-10-13 kernels 허브 우회 위험: Qwen3_5GatedDeltaNet은 use_kernel_forward_from_hub로 장식돼 있어 kernelize()가 forward를 통째로 교체하면 cache.update_recurrent_state 호출이 사라지고 state 훅이 조용히 우회된다. 설치된 소스 기준 교체는 from_pretrained(use_kernels=True) 또는 kernel_config 전달, model.use_kernels=True일 때만 일어나며 기본값은 False. 대응: scripts/check_hooks.py가 QuantStats를 layer_types와 대조해 PASS/FAIL 판정하고, FAIL이면 --no-hub-kernels(transformers import 전 USE_HUB_KERNELS=NO + use_kernels=False)로 재실행. 기본 매핑의 GDN 레이어 커널은 CUDA capability 121(GB10)과 ROCm 115에만 걸려 있어 L4/A100에서는 기본적으로 교체되지 않지만, 실험 전 매번 check_hooks로 확인한다.
- 2026-10-13 M1 검증을 Colab 대신 로컬 맥(M3 Pro 18GB, torch MPS)에서 수행. 이유: Colab은 로컬에서 원격 실행을 트리거할 공식 API가 없고, 0.8B는 MPS에 올라가며 순수 torch 폴백 커널이 그대로 돈다. tiny 모델로 CPU 대 MPS 수치 일치 확인(fp32 5e-7, bf16 2e-3). 2B 이상과 대규모 그리드는 Colab 유지.
- 2026-10-13 GPU별 dtype 정책: Ampere 이상(A100/H100/L4)과 MPS는 bf16, T4 등 compute capability 8.0 미만은 fp16, CPU는 fp32. hybridquant.resolve_dtype("auto")가 결정하며 HQ_DTYPE 환경변수로 덮어쓸 수 있다. 이유: T4는 bf16 텐서코어가 없어 bf16이 매우 느리거나 실패. 또한 transformers의 순수 torch 폴백은 fla/causal-conv1d가 "설치되지 않았을 때"만 동작하므로, T4에서는 두 패키지를 설치하지 않는다(노트북 GPU 선택 셀의 INSTALL_FAST_KERNELS). 결과 jsonl에 dtype을 기록해 bf16/fp16 결과가 섞이지 않게 한다.
- 2026-10-13 load_model MPS 우회: torch 2.14 + transformers 5.19에서 from_pretrained(device_map="auto")가 bf16이 아닌 dtype(fp16, fp32)을 MPS에 직접 올리면 가중치 로딩 중 crash(exit 139/134). fp16 계산 자체는 MPS에서 정상(CPU 로드 후 .to("mps")로 decode와 훅 발화 확인). 따라서 MPS이고 dtype이 bf16이 아니면 CPU에 로드 후 옮긴다. CUDA 경로는 변경 없음. 이 맥 전용 문제이며 T4/A100/H100과 무관.
