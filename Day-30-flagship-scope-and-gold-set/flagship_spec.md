# Flagship Spec: Tutorial Hell Navigator

**Author:** Kumbham Nikhil  
**Status:** In Progress (Day 30 Foundation)  
**Goal:** Help CS students move from passive tutorial watching to building and shipping real projects independently.

---

## 1. Why This Project?

| Criteria | Score | Reason |
| :--- | :--- | :--- |
| **Reachable Users** | High | CS classmates, peers, and developer communities. |
| **Corpus Defensibility** | High | Curated project blueprints, debugging failure trees, and MLOps checklists. |
| **Painful Problem** | High | Blank-editor paralysis, project abandonment, and production tooling confusion. |
| **Measurable Success** | High | 80-question benchmark with explicit technical pass/fail rubrics. |

---

## 2. Evaluation Benchmark (`gold_set.json`)

The system is evaluated against **80 real student questions** across 12 domains:
- Full-Stack & Project Milestones (`GS-01` – `GS-16`)
- Debugging, Docker & Tooling (`GS-17` – `GS-24`)
- AI & MLOps (`GS-25` – `GS-32`)
- Resumes & GitHub Portfolio (`GS-33` – `GS-40`)
- Study Habits & Fundamentals (`GS-41` – `GS-50`)
- DSA, Competitive Programming & OS/DBMS (`GS-51` – `GS-72`)
- Interviews & Assessments (`GS-73` – `GS-80`)

### Evaluation Rules:
* **Pass:** Must provide exact CLI commands, directory trees, Big-O stack derivations, or schema contracts.
* **Fail:** Generic advice ("keep practicing"), raw unparameterized SQL, or hallucinated APIs.

---

## 3. High-Level Architecture

1. **Input:** Student query, error log, or project roadblock.
2. **Retrieval:** Hybrid search (Qdrant dense vectors + BM25 sparse lexical matching) + Cross-Encoder reranking.
3. **Output:** Structured response with verified citations and actionable, step-by-step next actions.

> **Rule:** Every retrieval upgrade (hybrid search, rerankers, query rewriting) must improve the baseline score on the 80-question gold set to be kept.
