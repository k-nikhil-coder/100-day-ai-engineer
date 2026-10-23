# Day 30: Flagship Scoping & 80-Question Eval Gold Set

**Sprint:** 100-Day AI Engineer Sprint  
**Phase:** Phase 2 — Eval-First RAG & Real Users  
**Milestone:** Day 30 of 100  

---

## 🎯 What Was Built Today

1. **Flagship Project Scoped:** *Tutorial Hell Navigator for CS Students* — an evaluation-first AI mentor and retrieval system designed to guide students from passive tutorial copying to shipping verified, production-grade software.
2. **80-Question Gold Set (`gold_set.json`):** Curated 80 realistic student developer test cases across 12 core domains (full-stack, debugging, MLOps, DSA patterns, competitive programming, OS/networking fundamentals, and technical interview defense).
   - **Rubrics:** 2–3 required technical criteria per question (commands, layer orders, state transitions).
   - **Negative Constraints:** Strict automatic fails for vague encouragement ("keep practicing"), raw SQL string concatenation, or hallucinated APIs.
3. **Project Spec (`flagship_spec.md`):** Documented candidate scoring matrix, problem definition, and the 4-phase retrieval architecture.


---

## 📂 Repository Layout

```text
Day-30-flagship-scope-and-gold-set/
├── README.md              # Milestone summary and deliverables
├── flagship_spec.md       # Engineering RFC, target personas, and architecture spec
└── gold_set.json          # 80 evaluation test cases with rubrics and negative constraints
