"""Finish a run: wait for the runners, dedupe results/*.jsonl, regenerate every stage report and the summary,
copy results/ to Drive, commit and push. Meant to run in the background from the notebook:

    nohup python scripts/finish_run.py > results/finish.log 2>&1 &

GH_TOKEN is read from the environment. --no-wait fails instead of waiting if a runner is still alive.
"""
import os, re, json, glob, time, subprocess, datetime, sys, functools
print = functools.partial(print, flush=True)
WAIT = "--no-wait" not in sys.argv
# 결과(round1.log)가 있는 폴더를 실험 폴더로 씁니다. 커널이 중첩 clone 안에 들어가 있어도 안전합니다.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = next((d for d in (HERE, os.getcwd(), "/content/hybridquant")
             if os.path.exists(os.path.join(d, "results", "round1.log"))), None)
if REPO is None:
    raise SystemExit("no results/round1.log found: nothing to finish")
print("실험 폴더:", REPO)
DRIVE = "/content/drive/MyDrive/hybridquant_results"
KEYS = ("model", "label", "seed", "state_bits", "layer_idx", "context_len", "depth", "idx")

def runners():
    return subprocess.run(["pgrep", "-f", "scripts/run_(all|round|grid|niah|gsm8k)|layer_sweep"],
                          capture_output=True, text=True).stdout.split()

while WAIT and runners():
    print(datetime.datetime.now().strftime("%H:%M:%S"), "러너가 아직 돌고 있습니다. 60초 뒤 다시 확인합니다.")
    time.sleep(60)
if runners():
    raise SystemExit("러너가 아직 돌고 있습니다. 끝난 뒤 다시 실행하거나 WAIT=True 로 두십시오.")

# (1) 완료된 jsonl 의 중복 행 제거 (같은 run 이 두 번 기록된 경우 첫 행만 남깁니다)
print("\n[1] 중복 행 제거")
for p in sorted(glob.glob(f"{REPO}/results/*.jsonl")):
    seen, keep, n = set(), [], 0
    for l in open(p, encoding="utf-8"):
        if not l.strip(): continue
        n += 1; r = json.loads(l); k = tuple(r.get(x) for x in KEYS)
        if k not in seen: seen.add(k); keep.append(l if l.endswith("\n") else l + "\n")
    if len(keep) < n:
        open(p + ".tmp", "w", encoding="utf-8").writelines(keep); os.replace(p + ".tmp", p)
    print(f"  {os.path.basename(p):36s} {n:6d} -> {len(keep):6d}" + ("" if len(keep) == n else f"  ({n - len(keep)} 제거)"))

# (2) 시작된 모든 단계의 보고서와 그림, 전체 요약 재생성 (FAILED 로 끝났다가 이어 돌린 단계도 포함합니다)
print("\n[2] 보고서 재생성")
stages = []
for lg in ("round1.log", "round2.log"):
    p = f"{REPO}/results/{lg}"
    if os.path.exists(p):
        stages += [m.group(1) for l in open(p) for m in [re.search(r"=== step (\S+):", l)] if m]
for st in dict.fromkeys(stages):                     # 순서 유지, 중복 제거
    src = glob.glob(f"{REPO}/results/{st}*.jsonl")
    if not src: continue
    r = subprocess.run(["python", "analysis/report.py", "--stage", st, "--inputs", *src, "--out", f"results/reports/{st}.md"],
                       capture_output=True, text=True, cwd=REPO)
    print(f"  {st:20s} {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip()[-200:]}")
r = subprocess.run(["python", "analysis/report.py", "--stage", "all", "--inputs", *sorted(glob.glob(f"{REPO}/results/*.jsonl")),
                    "--out", "results/reports/summary.md"], capture_output=True, text=True, cwd=REPO)
print(f"  {'summary':20s} {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip()[-200:]}")

# (3) Drive 복사
if os.path.isdir(DRIVE):
    subprocess.run(["rsync", "-a", f"{REPO}/results/", f"{DRIVE}/"]); print("\n[3] Drive 복사 완료:", DRIVE)
else:
    print("\n[3] Drive 가 마운트돼 있지 않아 건너뜁니다")

# (4) 커밋과 push (토큰을 origin 에 다시 끼운 뒤 push 합니다). 먼저 로그에 찍힌 토큰을 가립니다: GitHub push protection 이 막습니다.
print("\n[4] 커밋과 push")
for lp in glob.glob(f"{REPO}/results/*.log"):
    txt = open(lp, encoding="utf-8", errors="replace").read()
    masked = re.sub(r"(github_pat_|ghp_|hf_)[A-Za-z0-9_]+", r"\1***", txt)
    if masked != txt:
        open(lp, "w", encoding="utf-8").write(masked); print("  masked tokens in", os.path.relpath(lp, REPO))
tok = os.environ.get("GH_TOKEN")
if tok:
    subprocess.run(["git", "-C", REPO, "remote", "set-url", "origin", f"https://{tok}@github.com/skytinstone/hybridquant.git"], check=True)
subprocess.run(["git", "-C", REPO, "add", "results"])
c = subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", "results: final (dedupe, regenerated reports and figures)"], capture_output=True, text=True)
print("  commit:", (c.stdout + c.stderr).strip() or "ok")
# code may have been merged into main meanwhile; rebase the results commits on top so the push is a fast-forward
pr = subprocess.run(["git", "-C", REPO, "pull", "--rebase", "-q", "origin", "main"], capture_output=True, text=True)
if pr.returncode != 0:
    print("  pull --rebase failed:", (pr.stdout + pr.stderr).strip()[-300:])
p = subprocess.run(["git", "-C", REPO, "push", "origin", "HEAD"], capture_output=True, text=True)
msg = (p.stdout + p.stderr).strip()
print("  push:", "ok" if p.returncode == 0 else "FAILED", (msg.replace(tok, "***") if tok else msg)[-300:])
print("\n", subprocess.run(["git", "-C", REPO, "status", "-sb"], capture_output=True, text=True).stdout.splitlines()[0])
