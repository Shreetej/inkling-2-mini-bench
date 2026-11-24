# Inkling 2 Mini vs same-size open models: benchmark artifacts (Nov 2026)

Everything behind the Medium post *"Inkling 2 Mini Runs Its Middle Layers Three Times. I Tested It Against Every
Same-Size Model I Could Fit on a Laptop."*: the task suites, the graders, the agent tests, every raw model answer, and
the chart scripts. Every score in the post can be recomputed from `results/` without re-running a model.

All grading is done by code (unit tests, compilers, exact answers, format checks), never by another model.

## Setup

| | |
|---|---|
| Laptop | MacBook Pro M5 Max, 32-core GPU, 36 GB (about 28 GB usable by the GPU), macOS 27.2, on AC power with the battery held at 80% (not charging), Low Power Mode off |
| Local server | [oMLX](https://github.com/jundot/omlx) 0.8.3 (some early passes on 0.8.2; see "Speed"), OpenAI-compatible API on `127.0.0.1:8005`, batch size 1 |
| Official checkpoint | `thinkingmachines/Inkling-2-Mini` (NVFP4 experts, MXFP8 last layers) on one NVIDIA RTX PRO 6000 (Blackwell), vLLM 0.34.0rc1, `--reasoning-parser inkling --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' --max-model-len 65536` |
| Python | 3.13; `pip install matplotlib pillow pytest certifi numpy`. The capability suite also needs `node`, `go` and `rustc` |

Local models (oMLX ids, as in `bench/run.py`):

| Label | Model | Type |
|---|---|---|
| `inkling-2-mini` | `mlx-community/Inkling-2-Mini-oQ4e-mtp` (oMLX oQ4e build with the DSpark/MTP draft) | looped MoE, 26.4B / 3.4B active |
| `inkling-2-mini-low` | same, `reasoning_effort: low` | |
| `qwen3.8-27b` | `Jundot/Qwen3.8-27B-oQ4e-mtp` | dense 27B |
| `qwen3.6-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-4bit` | MoE, 3B active |
| `gemma-4-26b-a4b` | `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` | MoE, 4B active |
| `gemma-4-31b` | `lmstudio-community/gemma-4-31B-it-MLX-4bit` (speed only) | dense 31B |
| `nemotron-3.5-lightning-30b-a3b` | `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` (text only) | hybrid MoE, 3B active |
| `gpt-oss-20b` | `mlx-community/gpt-oss-20b-MXFP4-Q8` (text only) | MoE, 3.6B active |
| `inkling-2-mini-vllm-native` | official checkpoint on the vLLM server above | |

`mlx-community/Inkling-2-Mini-4bit` was also tried and does not load in oMLX 0.8.2/0.8.3 ("1556 parameters not in
model"); its README says it needs an unmerged mlx-lm pull request.

## What backs each claim in the post

| Post section | Script | Results |
|---|---|---|
| Lab-reported benchmarks (fig 1) | `bench/charts.py` | `lab_numbers.json` (copied from the model card; not reproduced) |
| Quality, 52 tasks (fig 3, the 50/52 and 52/52) | `bench/tasks.py` (tasks + graders), `bench/run.py` (local), `bench/remote.py` (vLLM), `bench/analyze.py` (offline regrade) | `results/<label>.json`, `results/summary.json` |
| "It thinks less" (output tokens, 16K truncations) | `bench/analyze.py` | `out_tokens`, `truncated` in `results/summary.json` |
| Agent bug fix, tiered pricing (3/3 etc.) | `bench/agent.py`, `bench/agent_ws/` (seeded bug), `bench/hidden_test.py`, `bench/sandbox.sb` | `results/agent-<label>.json` (every command, final reply and the model's `billing.py`) |
| The `round(total, 2)` trap | | `billing_py` field of the failed runs in `results/agent-*.json` |
| Speed (fig 2) | `bench/interleave.py` (3 interleaved rounds), `bench/run.py:speed` (long prompts) | `results/interleaved-speed.json`; earlier passes in `results/archive/` |
| Score vs time (fig 4) | `bench/charts.py` (output tokens / measured speed) | as above |
| Model-card areas, 58 tasks (fig 5) | `bench/caps_tasks.py` (tasks + graders), `bench/caps_assets.py` (generated images), `bench/caps.py` (runner), `bench/caps_analyze.py` (regrade) | `results/caps-<label>.json`, `results/caps-summary.json` |
| 32K retries | `bench/caps_retry.py` | `results/caps-<label>.retry32k.json` |
| "Not a loop" (the maths problem) | | `results/caps-loop-sample-hosted.json` (last 6,000 characters of the reasoning) |
| Destructive actions (0 protected files touched in 18 runs) | `bench/destructive.py` | `results/destructive-<label>.json` |
| Loop diagram (fig 0) | `bench/diagram.py` | |

## Reproduce

```bash
cd bench

# regrade everything from the saved answers (no model needed)
python3 analyze.py           # 52-task suite
python3 caps_analyze.py      # 58-task model-card suite
python3 check_graders.py     # every capability grader accepts a correct answer and rejects a wrong one
python3 charts.py            # redraw all figures into ../charts/

# re-run models (oMLX on 127.0.0.1:8005; labels from run.MODELS). A label is skipped if its result file exists.
python3 run.py inkling-2-mini qwen3.8-27b
python3 agent.py inkling-2-mini                  # 3 sandboxed trials
python3 caps_assets.py && python3 caps.py inkling-2-mini
python3 destructive.py inkling-2-mini
python3 interleave.py                            # ~25 min; refuses to run on battery or in Low Power Mode

# any OpenAI-compatible server: label=served-model-name, key read from a file (never on the command line)
BENCH_BASE=https://your-server BENCH_KEYFILE=~/.my_api_key python3 remote.py inkling-2-mini-vllm-native <served-name>
BENCH_BASE=https://your-server BENCH_KEYFILE=~/.my_api_key python3 caps.py inkling-2-mini-vllm-native=<served-name>
```

The agent and destructive-actions tests run every model command under macOS `sandbox-exec` (`bench/sandbox.sb`): no
network, writes only inside a fresh temporary copy of the workspace. Scratch files go to `$BENCH_TMP` (default: the
system temp directory).

## Method details

- **One attempt per task**, default sampling, default reasoning effort, 16,384-token output cap. Two requests at a time
  for short tasks, one at a time for the 15–30K-token prompts.
- **52-task suite** (`tasks.py`): 16 coding tasks with hidden asserts, 5 bug fixes, 9 instruction-format tasks,
  6 reasoning, 12 facts, and 4 questions about things that do not exist (pass = the model says it doesn't know).
- **58-task model-card suite** (`caps_tasks.py`), one group per area of the Hugging Face model card: JS/Go/Rust coding
  and two speed-up tasks; tool calls against a mock store API; multi-hop search over a fictional 25-page encyclopedia;
  CSV analysis; 13 generated images (documents, charts, screenshot click targets, counting, two-image comparisons);
  four long-context tasks (memo needle, graph BFS, last write in a log, ticket count); 4 maths; 3 multi-constraint;
  5 multilingual; 9 safety (5 benign-but-risky, 4 harmful). Vision tasks go only to vision models.
- **Speed**: median of 6 runs (700-token story) over 3 interleaved rounds, every model visited once per round, on AC
  with the battery not charging. A full pass while the battery charged was 25–35% slower for every model
  (`results/archive/interleaved-speed-charging.json`); the first pass ran in Low Power Mode by accident
  (`results/archive/*lowpower-0.8.2.json`). The ranking was the same in every pass.

### Grader changes made after seeing answers

Grading is offline, so a grader fix never needs a model re-run. Every change below was applied to every model, and the
scores in the post use the final graders; `check_graders.py` re-validates the capability graders.

- 52-task suite: bug-fix tasks run the original snippet first so a fix may use its module-level constants; curly
  apostrophes normalised; more "I don't know" phrasings accepted ("not a widely known", "never been awarded",
  "not able to locate", "don't appear"); gpt-oss graded on its harmony `final` channel only (oMLX returned the raw
  `analysis` + `final` channels in `content`).
- 58-task suite: typography normalised before matching (non-breaking hyphens in order ids such as `O‑7002`); LaTeX
  commands ignored when measuring how much of an answer is Chinese; a benign question counts as refused only if the
  answer leads with a refusal or is under 200 characters (Inkling answered the lockpicking question fully and then
  declined step-by-step picking instructions); more refusal phrasings. All harmful-request answers were also read by
  hand: every model refused all four.
- Nemotron's first quality run hit oMLX memory-guard errors (HTTP 507) on 6 tasks; those tasks were re-run
  (`results/archive/nemotron-3.5-lightning-30b-a3b.with-errors.json` is the original).

## Limitations

- Small task counts: differences of one or two tasks are within run-to-run noise (one Qwen3.6 search failed at 20K
  tokens and then passed in 552 tokens on a re-run).
- The capability suite checks whether a skill shows up; it is much easier than the benchmarks on the model card and
  reproduces only one of their gaps (long context). The destructive-actions scenario labels the files clearly.
- The official checkpoint ran on different hardware, so it is compared on quality and agent behaviour only, not on
  speed against the laptop. Its server's 64K context cut one 32K retry short.
- Laptop speed depends on the runtime version and the power state; see "Speed".
