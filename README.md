# ProbeFL Artifact

This repository contains the code, data, and scripts for the paper:
**"Probe First, Guess Later: Empowering LLMs with Interactive Debuggers for Method-Level Fault Localization."**

## Contents

```
├── code/                  # Core implementation
│   ├── autofl.py          # Main entry point
│   ├── run_probefl_batch.py  # Batch experiment runner
│   ├── compute_score.py   # Scoring script
│   ├── lib/               # Core libraries
│   │   ├── protocol.py    # JDB backend + ablation logic
│   │   ├── verify_agent.py    # JDB verification agent
│   │   ├── llm_utils.py   # LLM API client (supports OpenAI-compatible endpoints)
│   │   ├── repo_interface.py  # Repository abstraction layer
│   │   ├── d4j_interface.py   # Defects4J-specific interface
│   │   └── ...
│   └── prompts/           # Agent prompts
├── scripts/               # Analysis scripts
│   ├── rq1/               # RQ1: comparison with baselines
│   ├── rq2/               # RQ2: ablation study
│   └── rq3/               # RQ3: cost & runtime analysis
├── data/baseline/         # Baseline comparison data (CSV)
├── figures/               # Paper figures (PDF + LaTeX tables)
└── scores/                # Final per-project scoring results (JSON + CSV)
```

## Environment Setup

```bash
# Python 3.10+
pip install openai python-dotenv javalang pandas numpy tqdm matplotlib venn

# Defects4J v1.2.0 must be installed and available at $D4J_HOME
# The project expects Defects4J checkouts at: data/defects4j/{bug_name}/
```

## Configuration

Create a `.env` file with your API credentials:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE_URL=https://api.openai.com/v1
```

The code is compatible with any OpenAI-compatible API endpoint.

## Running a Single Bug

```bash
python3 code/autofl.py \
  -m gpt-5.4-2026-03-05 \
  --verifier_model gpt-5.4-2026-03-05 \
  -b Math_2 \
  -p code/prompts/system_msg_newfl.txt \
  -o results/Math_2.json \
  --max_budget 10 \
  --show_line_number \
  --postprocess_test_snippet \
  --allow_multi_predictions \
  --dynamic
```

## Running Batch Experiments

```bash
# FULL mode
python3 code/run_probefl_batch.py \
  --model gpt-5.4-2026-03-05 \
  --verifier_model gpt-5.4-2026-03-05 \
  --base_label ProbeFL_54 \
  --projects all \
  --budget 10 \
  --timeout 2400 \
  --prompt code/prompts/system_msg_newfl.txt \
  --parallel 2 \
  --modes full

# Ablation modes: no_context_bridge, no_reflection, no_dynamic
```

## Reproducing Paper Results

### RQ1: Comparison with Baselines

```bash
# Step 1: Generate per-project scores
for proj in Chart Closure Lang Math Time; do
  PYTHONPATH=. python3 scripts/rq2/compute_score_verifl.py \
    results/ProbeFL_54_FULL/gpt-5.4-2026-03-05 \
    --language java -a -p $proj \
    -o scores/${proj}_ProbeFL_54_FULL_scores.json
done

# Step 2: Run comparison
PYTHONPATH=. python3 scripts/rq1/run_rq1.py --policy common
PYTHONPATH=. python3 scripts/rq1/plot_rq1.py --policy common
```

### RQ2: Ablation Study

```bash
python3 scripts/rq2/run_rq2.py \
  --score-dir scores/ \
  --out-dir scores/
```

### RQ3: Cost & Runtime Analysis

```bash
# Step 1: Summarize results
PYTHONPATH=. python3 scripts/rq3/summarize_probefl_rq3_by_project.py \
  --input-dir results/ProbeFL_54_FULL/gpt-5.4-2026-03-05 \
  --output-prefix scores/probefl_rq3_full \
  --mode-name ALL \
  --exclude-errors

# Step 2: Generate figures and tables
PYTHONPATH=. python3 scripts/rq3/plot_rq3_overhead_boxplot.py \
  --per-bug scores/probefl_rq3_full_per_bug_details.csv
PYTHONPATH=. python3 scripts/rq3/plot_rq3_dynamic_tool_violin.py \
  --per-bug scores/probefl_rq3_full_per_bug_details.csv
PYTHONPATH=. python3 scripts/rq3/analyze_rq3_cost.py \
  --per-bug scores/probefl_rq3_full_per_bug_details.csv
PYTHONPATH=. python3 scripts/rq3/summarize_rq3_probing_invocation.py \
  --per-bug scores/probefl_rq3_full_per_bug_details.csv
```

## Model Configuration

All experiments were conducted using:
- **Main Agent**: `gpt-5.4-2026-03-05`
- **Verifier Agent**: `gpt-5.4-2026-03-05`
- **Temperature**: 0.0
- **Max Budget**: 10 steps per bug
- **Max Verification Attempts**: 2 (1 in no_reflection ablation)

The API endpoint used was an OpenAI-compatible proxy. Any OpenAI-compatible endpoint
supporting `gpt-5.4-2026-03-05` or equivalent models can be used.

## Key Results (353 Defects4J v1.2.0 bugs)

| Metric | ProbeFL | PingFL | AutoFL | Ochiai |
|--------|---------|--------|--------|--------|
| acc@1 | 284 (80.5%) | 242 (70.1%) | 245 (71.0%) | 102 (29.6%) |
| acc@3 | 303 (85.8%) | 273 (79.1%) | 264 (76.5%) | 185 (53.6%) |
| MRR | 0.855 | 0.746 | 0.735 | 0.447 |

### Ablation (acc@1)

| Mode | acc@1 | Δ vs FULL |
|------|-------|-----------|
| FULL | 284 | — |
| w/o Context Bridge | 278 | -6 |
| w/o Reflection | 267 | -17 |
| w/o Dynamic | 262 | -22 |

### Cost per Bug

| Stage | Tokens | Cost (USD) |
|-------|--------|------------|
| Static Search | 47.8K | $0.156 |
| Runtime Probing | 167.9K | $0.482 |
| **Total** | **215.6K** | **$0.638** |
