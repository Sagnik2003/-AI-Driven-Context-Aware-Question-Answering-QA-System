

# **README.md — Disaster-Response AI Pipeline**


# Disaster-Response AI Pipeline

A modular, interpretable and safety-first AI system designed to answer disaster-related citizen queries  
(water safety, compensation, health, pest control).  
The architecture is fully explainable and avoids hallucination by grounding all answers  
in official knowledge snippets.

Each module performs a distinct cognitive function:
- **Bayesian Intent Reasoning**
- **Search-Based Retrieval (BFS + A\*)**
- **Planning (GraphPlan + POP)**
- **Reinforcement Learning Policy (Q-learning)**
- **LLM-Based Final Answer Generation**

---

## 🚨 Overview

This system transforms a raw user query into a **safe, grounded, and well-structured final answer**.

```

User Query
↓
Module A — Bayesian Reasoning
↓
Module B — Retrieval (BFS & A*)
↓
Module C — Planning (GraphPlan + POP)
↓
Module D — RL Decision Policy
↓
Module E — LLM Response Generator (grounded in KB)

```

The pipeline ensures:
- ✔ no hallucinations  
- ✔ transparent decision-making  
- ✔ safety escalation for high-risk queries  
- ✔ complete traceability (posterior, search scores, actions, RL choices, snippet provenance)

---

## 📂 Project Structure

```

.
├── search_retrieval.py        # Module B – KB, BFS, A*, TF-IDF, retrieve_with_bn()
├── planning_module.py         # Module C – GraphPlan + POP planner
├── rl_policy.py               # Module D – Q-learning + RL decision logic
├── module5_llm.py             # Module E – LLM prompt builder and final generator
├── bayesian_intent.py         # Module A – feature extraction + Bayesian posterior
├── AI_project(theory).pdf     # Official KB source (optional; referenced as provenance)
├── demo_plans.json            # Output from demo runs (optional)
└── README.md                  # (this file)

````

---

## 🧩 Module Breakdown

### **Module A — Bayesian Intent Reasoning (`bayesian_intent.py`)**
Extracts features:
- QueryType (Water / Compensation / Agriculture / Health / Other)
- Keyword Ambiguity
- Urgency Level
- Location Type

Then computes:
- Posterior P(Intent | Observed Features)

**Used by:** Retrieval, Planner, RL.

---

### **Module B — Search Retrieval (`search_retrieval.py`)**
Contains:
- Knowledge Base (KB_SNIPPETS)
- Graph adjacency between snippets
- TF-IDF vectorizer + similarity matrix

Algorithms:
- **BFS Search** — semantic exploration of KB graph  
- **A\* Search** — uses `1 – cosine similarity` as heuristic  
- **Intent-aware scoring** using posterior from Module A

**Output:**  
Chosen snippet + BFS candidates + A\* candidates + scores + provenance.

---

### **Module C — Planning (`planning_module.py`)**
Implements:

#### 1. **GraphPlan**
- Layered forward progression with actions & propositions  
- Produces action layers and proposition layers  
- Produces an initial linear candidate plan

#### 2. **POP (Partial-Order Planner)**
- Generates a valid partial ordering of actions  
- Constructs dependencies based on preconditions/effects  
- More flexible than linear GraphPlan

Planner adapts based on:
- Risk level  
- Posterior confidence  
- Intent category  
- RL recommendations  

**Output:**  
Structured plan steps with metadata, retrieval references, and safety requirements.

---

### **Module D — Reinforcement Learning Policy (`rl_policy.py`)**
Implements a small Q-learning agent on a discrete state space:

#### State =  
(intent_index, confidence_bin, location_bin, risk_bin)

#### Actions:
- 0 → DirectAnswer  
- 1 → AskClarification  
- 2 → EscalateToHuman  
- 3 → GenerateExplanation  

RL learns:
- When to clarify  
- When to escalate  
- When direct answer is safe  
- When explanations are better  

RL output can modify planner steps dynamically.

---

### **Module E — LLM Final Answer Generation (`module5_llm.py`)**
Constructs a **safety-first prompt** using:
- User query  
- Posterior intent  
- Context (location, urgency, risk)  
- Retrieved snippet  
- Planner sequence  
- RL decision rationale  

LLM Backends:
- **OpenAI API** (if API key available)  
- **HuggingFace Transformers** (local model)  
- **Mock LLM** (offline fallback, deterministic)  

Enforces strong grounding:
- If LLM output contradicts official snippet → auto-corrects  
- If snippet inadequate → system safely admits limitations

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install scikit-learn numpy pandas
````

Optional packages:

```bash
pip install openai
pip install transformers torch
```

---

## ▶️ Running Individual Modules

### **1. Retrieval Test**

```bash
python search_retrieval.py
```

### **2. Planner Demo**

```bash
python planning_module.py
```

### **3. RL Policy Demo**

```bash
python rl_policy.py
```

### **4. LLM Generator Demo**

```bash
python module5_llm.py
```

---

## 🧪 Programmatic Example

```python
from bayesian_intent import extract_features, posterior_intent
from search_retrieval import retrieve_with_bn
from planning_module import build_plan
from rl_policy import RLPolicy
from module5_llm import generate_final_answer

query = "Is it safe to drink water here?"

# 1. Bayesian reasoning
features = extract_features(query)
posterior = posterior_intent(features)

# 2. Retrieval
retrieval = retrieve_with_bn(query)

# 3. Planning
plan = build_plan(features, posterior, retrieval)

# 4. RL refinement
policy = RLPolicy()
rl_decision = policy.decide_action_for_plan_result(plan)

# 5. Final answer
response = generate_final_answer(plan)
print(response["final_text"])
```

---

## 🔐 Safety Guarantees

This system ensures:

✔ **No hallucinated safety advice**
✔ **Official KB is the only source of truth**
✔ **Grounded responses only**
✔ **Clarification for low-confidence queries**
✔ **Escalation for high-risk situations**
✔ **Full provenance tracking**

---

## ❗ Troubleshooting

**`ImportError: search_retrieval.py not found`**
→ Ensure the retrieval module is in the project directory.

**LLM output not grounded**
→ Module E will automatically correct it using the snippet.

**Planner produces unexpected steps**
→ Adjust thresholds in the planner or RL reward logic.

---

## 🚀 Future Extensions

* Real-time integration with municipal alert systems
* Larger KB from PDFs + vector database
* Human-in-the-loop RLHF
* FastAPI web deployment
* Android app using lightweight on-device LLM

---

## 📜 License

For research and educational purposes only.
Review safety and policy requirements before production deployment.



