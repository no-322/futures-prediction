# Futures Price Prediction — Minute-Bar Direction Classifier

One-line description: predict next-minute up/down direction on
futures data using lagged OHLCV+VWAP features and three classical
models (RF, GBM, SVM), benchmarked against simple baselines.

## TL;DR


## Methodology

### Natural Language Coding as a deliberate choice
This project was built using Claude Code in a "vibe coding"
workflow — natural-language prompts produce Python implementations
guided by CLAUDE.md and a `feature-engineering` skill. Every prompt
I submitted is logged to `prompts/log.md` via a UserPromptSubmit
hook, making the reasoning trail version-controlled alongside the
code. 

This addresses a common limitation of LLM-assisted work: the
reasoning trail is usually invisible. Logging prompts doesn't
make outputs deterministic - sampling stochasticity, model
drift, and context sensitivity mean the same prompt may yield
slightly different code on a different day. But it makes the
*methodology auditable*. The trail of decisions, iterations,
and clarifying questions becomes version-controlled alongside
the code, the way a lab notebook does in physical sciences.
The code itself is fully reproducible (fixed seed, deterministic
pipeline); the prompts document how it came to exist.

### Feature engineering and labeling
20-dim lagged vector built from the 4 minutes prior to each
target minute. Full spec: [CLAUDE.md -> Feature specification](CLAUDE.md#feature-specification).
The construction logic is auto-loaded from the
[feature-engineering skill](.claude/skills/feature-engineering/SKILL.md).


### Train/test discipline
50/50 time-ordered split, no shuffle. Rules enforced in
[CLAUDE.md → Non-negotiable rules](CLAUDE.md#non-negotiable-rules)
and verified by [tests/test_no_leakage.py](tests/test_no_leakage.py).

### Models


## Results


## Reproducing
```bash
git clone ...
pip install -e .
pytest                    # all tests pass
python -m src.models.rf   # train RF
python -m src.evaluate    # full comparison
```

## Repo structure
.
├── CLAUDE.md
├── .claude/
│   ├── settings.json        
│   ├── scripts/             
│   └── skills/              
├── data/
│   ├── raw/                 
│   └── processed/           
├── src/
│   ├── load.py              
│   ├── split.py             
│   ├── features.py         
│   ├── labels.py            
│   ├── evaluate.py          
│   └── models/{rf,gbm,svm}.py
├── tests/                   
├── prompts/log.md           
├── docs/
│   ├── MODULES.md           
│   └── notes/               
└── pyproject.toml


## Theoretical grounding
Notes on each algorithm from Elements of Statistical Learning:
- `docs/notes/evaluation.md` — Ch 7 (model assessment)
- `docs/notes/rf.md` — Ch 15 (Random Forests)
- `docs/notes/gbm.md` — Ch 10 (Boosting)
- `docs/notes/svm.md` — Ch 12 (SVMs)

## Honest limitations


## What I'd do differently
