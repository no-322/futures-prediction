# Prompt Log

## Bootstrap (pre-logging)
2026-05-24 — Initial scaffold (CLAUDE.md, .claude/settings.json, hook scripts,
directory structure) created manually from a Claude.ai conversation, before
the UserPromptSubmit hook was active. From the skill file onward, all prompts
are auto-logged.

---
## 2026-05-24 15:16:00

Create pyproject.toml for this project. Python 3.11+. Dependencies: pandas, numpy, scikit-learn, xgboost. Dev extras: pytest, ipython. Configure pytest to use the tests/ directory.

---
## 2026-05-24 15:22:47

stage and commit with appropriate message

---
## 2026-05-24 16:09:33

I need to create a skill file .claude/skills/feature-engineering/SKILL.md as a skill that auto-triggers whevever features are being built, modified or tested. Include YAML frontmatter with name and a description that's specific enough to trigger reliably. The body should cover: when to apply, the 5-feature × 4-lag (20-dim) construction rule from CLAUDE.md, the no-leakage rule (minute t never appears in features for row t, watch out for pandas rolling() defaults), verification steps before declaring done (pytest, shape check, MODULES.md update), and common mistakes for time-series feature engineering.

---
## 2026-05-24 16:24:05

stage and commit with appropriate changes

---
## 2026-05-24 16:24:26

stage and commit with appropriate changes

---
## 2026-05-24 16:38:26

Read data/raw/data.csv. Don't write production code yet — just inspect the data. Then propose an update to the "Data" section of CLAUDE.md (add the section if it doesn't exist yet) covering: file location, schema (columns with dtypes), row count, time range, sort order, and any quirks (NaNs, zero-volume rows, gaps, duplicates). Show me the proposed change before applying.

---
## 2026-05-24 16:52:45

stage and commit appropriate message. Add docs/notebook.md as well. It contains my learnings based on the prompts. It is like a lab note

---
## 2026-05-25 12:12:17

I need to load the data next. The data is present in data/raw/*.csv. Write tests to validate schema ensuring the columns needed to construct the feature vector is present


---
## 2026-05-25 12:44:43

stage and commit with appropriate message

---
## 2026-05-25 13:18:14

I need to write src/split.py next. This function splits the data loaded in by src/load.py into train and test sets. the first 50% of data based on timestamp would be the train set and the next 50% would be test set. Ensure that the split is reproducible and that train and test datasets are non-overlapping

---
## 2026-05-25 13:25:01

stage and commit with appropriate message

---
## 2026-05-25 13:36:12

We need to modify load.py so that print warning is not the only action taken when date time is not monotonically increasing. We should sort rows if it is not, so that load.py does not provide garbage results.

---
## 2026-05-26 20:16:17

stage and commit with relevant comments. Updated notebook.md to reflect learning on Monday 25th May. Added file evaluation.md- it contains notes of ESL topics 7.1-7.4

---
## 2026-05-26 20:16:46

stage and commit with relevant comments. Updated notebook.md to reflect learning on Monday 25th May. Added file evaluation.md- it contains notes of ESL topics 7.1-7.4

---
## 2026-05-26 20:53:17

Next we write src/features.py. This reads the training data provided by split.py and creates the 20 dimensional feature vector. If any of t-1, t-2, t-3 or t-4 data is empty, fill in value from previously available row from the same column. Drop first 4 rows to remove null values. Write test cases to match row-count post-transformation+4 = train data count  and test cases to match dimension of columns as well and ensure there are no null values

---
## 2026-05-26 22:49:43

stage and commit all changes with relevant comments
