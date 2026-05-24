# Project Notebook

## 2026-05-24 — Sunday

**Built** the scaffolding and foundation. I learnt proper usage of hooks and set up log files to record userprompts. Initially I assumed this would lead to deterministic action. But later realized the model responses are stochastic in nature. Although the log file doesn't guarantee replicability, it does give an audit trail that shows the thought process as well. 

Added [CLAUDE.md](./../CLAUDE.md) file to ensure that the methodology is specified and to avoid common traps during the machine learning process.

I caught claude hallucinating what the tick columns represents. Although it seemed reasonable enough, assumption on input data is dangerous.

Did some basic EDA with claude as well. An interesting point that came up is for some records there is a gap of more than 1 minute - 2 min, 3min, overnight, weekends, etc. *How do we handle this?* On one hand we could just consider the previous record. But even if market is not live there would have been changes to the underlying and capturing previous record might give us stale data. 