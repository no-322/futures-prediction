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
