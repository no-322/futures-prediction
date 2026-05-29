# Project Notebook

## 2026-05-24 — Sunday

**Built** the scaffolding and foundation. I learnt proper usage of hooks and set up log files to record userprompts. Initially I assumed this would lead to deterministic action. But later realized the model responses are stochastic in nature. Although the log file doesn't guarantee replicability, it does give an audit trail that shows the thought process as well. 

Added [CLAUDE.md](./../CLAUDE.md) file to ensure that the methodology is specified and to avoid common traps during the machine learning process.

I caught claude hallucinating what the tick columns represents. Although it seemed reasonable enough, assumption on input data is dangerous.

Did some basic EDA with claude as well. An interesting point that came up is for some records there is a gap of more than 1 minute - 2 min, 3min, overnight, weekends, etc. *How do we handle this?* On one hand we could just consider the previous record. But even if market is not live there would have been changes to the underlying and capturing previous record might give us stale data. 

## 2026-05-25 — Monday

*Built* src/load.py, src/split.py. The shell files ensure that anytime a file in src is run, the relevant test files are autotriggered. Hence, first we create the test files. Then the src files and run it. 

*Mistake noticed:* I noticed that claude just prints a warning when datetime is not in monotonic order. But the downstream file(load.py) runs with it assuming it is not monotonic. So I changed it to ensure it is always sorted. I first tried it in `accept edits on` mode but it started to assert error when datetime was out of error. So reset to previous commit and did it in plan mode verifying the changes. 

## 2026-05-26 - Tuesday
In addition to this I also want a deeper understanding of everything I am coding, so I am currently studying Elements of Statistical Learning sections 7.1-7.4. 

*Built* features.py and labels.py to create features and labels. *Mistake Noticed:* Despite mentioning in [CLAUDE.md](../CLAUDE.md) claude failed to update function signatures in [MODULES.md](MODULES.md)

## 2026-05 - Wednesday
I realised I made a mistake. I had the order `load` -> `split` -> `feature extraction` -> `label`. But this pipeline would need a rerun as test dataset would need a rerun of `feature extraction`. So changed the pipeline order to `load` -> `feature extraction` -> `split`  -> `label`

## 2026-05 - Thursday

HTML was added to log.md. Seems to be a bug. I need to figure out why this is happening. I also learnt about joblib model persistence and how we wouldn't have to retrain everytime, this will be handy for more complex models like SVM. So I am going to add it for the other models as well. 

Turns out Claude injects XML into user message stream when a background task is run. When evaluation.py was written a background task of run all tests was trigerred. So this was logged in as well. It is interesting to note that while this is being logged, clarification questions asked post plan or during edit are not logged. 