# 🧠 Self-Healing Text Classification Pipeline

> **A local DistilBERT model that knows when it's unsure — and automatically calls a free LLM to fix itself.**  
> Built with LangGraph · DistilBERT · Groq (Llama 3.1) · HuggingFace Transformers

---

## 📌 Table of Contents

1. [Introduction — What Is This?](#-introduction--what-is-this)
2. [The Problem It Solves](#-the-problem-it-solves)
3. [Why I Built This](#-why-i-built-this)
4. [Architecture & Approach](#-architecture--approach)
5. [Why This Approach?](#-why-this-approach)
6. [Model Used & Accuracy](#-model-used--accuracy)
7. [Test Scores & Metrics](#-test-scores--metrics)
8. [Model Comparison](#-model-comparison)
9. [Benefits of This System](#-benefits-of-this-system)
10. [Dependencies — Exact Versions](#-dependencies--exact-versions)
11. [🔑 Getting Your Free Groq API Key](#-getting-your-free-groq-api-key)
12. [Step-by-Step Setup Guide](#-step-by-step-setup-guide)
13. [How to Run](#-how-to-run)
14. [Project Structure](#-project-structure)
15. [Estimated Setup Time](#-estimated-setup-time)
16. [Troubleshooting](#-troubleshooting)

---

## 🌟 Introduction — What Is This?

This project is a **Self-Healing Text Classification Pipeline** — a smarter, fault-tolerant AI system that classifies text as **POSITIVE** or **NEGATIVE** (sentiment analysis), but unlike standard classifiers, it has a built-in rescue mechanism.

Here is what makes it special:

1. A **fine-tuned DistilBERT model** runs locally and makes the first prediction
2. The system **checks its own confidence score** — is it really sure?
3. Based on how confident it is, it automatically takes one of three paths:
   - ✅ **High confidence (>80%)** → Accept the result immediately
   - ⚠️ **Medium confidence (50–80%)** → Flag it for human review
   - 🚨 **Low confidence (<50%) or Error** → Send to an LLM "Self-Healer" (Groq/Llama) to recover

This is orchestrated using **LangGraph** — a graph-based workflow framework — making the entire routing logic transparent, visual, and easily extensible.

---

## 🔥 The Problem It Solves

Standard machine learning classifiers have a critical blind spot: **they always give you an answer, even when they have no idea.**

Imagine deploying a sentiment classifier in production. When it's uncertain, it still silently picks a label with low confidence and you never know. This causes:

- Silent misclassifications in real applications
- No fallback when the model fails on unusual inputs
- No way to distinguish "the model is confident" vs "the model is guessing"

### The Gap vs This Solution

| Standard Classifier | This Self-Healing Pipeline |
|---|---|
| Always returns a label, even when uncertain | Checks confidence before accepting |
| Crashes or returns garbage on errors | Catches errors and routes to fallback |
| No recovery mechanism | LLM healer fixes low-confidence predictions |
| Binary pipeline — one path | Three-path routing based on confidence |
| No audit trail | Full state logged at every step |

---

## 💡 Why I Built This

I wanted to explore a real-world problem in AI deployment: **what happens when your model isn't sure?**

Most tutorials show you how to train a model and run it. Almost none show you how to build a system that handles uncertainty intelligently. I built this to:

1. Learn how **LangGraph** works for orchestrating multi-node AI workflows
2. Explore the concept of **confidence-based routing** in ML pipelines
3. Demonstrate a practical pattern of using a large LLM only as a last resort (cost-efficient)
4. Build something useful for anyone learning NLP + LangChain tooling together

**Real use cases this pattern applies to:**
- Customer review sentiment analysis (e-commerce)
- Medical text triage (flag uncertain cases for doctor review)
- Content moderation pipelines
- Any production ML system that needs graceful degradation

---

## ⚙️ Architecture & Approach

### How the Pipeline Flows

```
         [ User Input Text ]
                  ↓
     ┌────────────────────────┐
     │  Node 1: Local         │  ← DistilBERT (runs on your machine)
     │  Predictor             │
     └────────────────────────┘
                  ↓
         [ Confidence Router ]
         ┌────────┼─────────┐
         ▼        ▼         ▼
      > 0.80   0.50–0.80   < 0.50 or Error
         ↓        ↓         ↓
      ACCEPT   CLARIFY   FALLBACK
         ↓        ↓         ↓
        END    Human     Node 2:
               Review    Self-Healer LLM
                ↓        (Groq / Llama 3.1)
               END           ↓
                            END
```

### What Each Component Does

| Component | File | Role |
|---|---|---|
| `local_predictor_node` | `pipeline/langgraph_pipeline.py` | Runs DistilBERT, returns prediction + confidence |
| `decision_logic` | `pipeline/langgraph_pipeline.py` | Router — checks confidence and decides path |
| `clarify_node` | `pipeline/langgraph_pipeline.py` | Flags medium-confidence results for human review |
| `fallback_node` | `pipeline/langgraph_pipeline.py` | Calls Groq LLM to recover low-confidence/failed predictions |
| `GraphState` | `pipeline/langgraph_pipeline.py` | Shared state object passed between all nodes |

---

## 🤔 Why This Approach?

### Why LangGraph?

LangGraph was chosen over simpler alternatives because:

| Alternative | Problem |
|---|---|
| Plain `if/else` in Python | Works, but no graph state, no streaming, no visualisation |
| LangChain LCEL chains | Linear only — cannot loop or branch based on runtime state |
| Manual threading/async | Complex to build and debug |
| **LangGraph StateGraph** ✅ | Clean node-edge model, built-in conditional routing, state management, and visual graph generation |

### Why DistilBERT for the Local Model?

- **Fast** — 60% smaller and 2× faster than BERT with 97% of the performance
- **Proven** — 91.3% accuracy on SST-2 sentiment benchmark
- **Free** — Available on HuggingFace, no API cost
- **Local** — Runs entirely on your machine, no internet needed for prediction

### Why Groq as the Fallback LLM?

- **Free tier available** — No credit card required to start
- **Extremely fast inference** — Groq's LPU hardware makes Llama 3.1 respond in under 1 second
- **Only used when needed** — The LLM fallback is triggered only for hard/uncertain cases, keeping API usage minimal

---

## 🤖 Model Used & Accuracy

**Local Model:** `distilbert-base-uncased-finetuned-sst-2-english`  
**Source:** [HuggingFace Model Card](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)  
**Dataset:** GLUE SST-2 (Stanford Sentiment Treebank)  
**Task:** Binary Sentiment Classification (POSITIVE / NEGATIVE)

---

## 📊 Test Scores & Metrics

These metrics are from the official DistilBERT SST-2 evaluation on the GLUE validation set:

### Performance Scores

| Metric | Score |
|---|---|
| ✅ Accuracy | **91.3%** |
| 🎯 Precision | **89.7%** |
| 🔁 Recall | **93.0%** |
| ⚖️ F1 Score | **91.3%** |

### Visual Score Breakdown

```
Model Performance on GLUE SST-2 Validation Set
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accuracy    █████████░  91.3%
Precision   ████████░░  89.7%
Recall      █████████░  93.0%
F1 Score    █████████░  91.3%

            0%  20%  40%  60%  80%  100%
```

### Pipeline Routing Behaviour (4 Test Cases)

```
Test Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test 1 (High Confidence >80%)   ✅ ACCEPTED   → Direct result, no fallback needed
Test 2 (Medium Confidence ~65%) ⚠️  CLARIFY    → Routed to human review
Test 3 (Low Confidence ~30%)    🔄 FALLBACK   → LLM healer corrected prediction
Test 4 (Input too long / Error) 🔄 FALLBACK   → LLM healer recovered from crash
```

---

## 📈 Model Comparison

How DistilBERT compares to similar sentiment classification models on SST-2:

| Model | Accuracy | Speed (CPU) | Size | Cost | Local? |
|---|---|---|---|---|---|
| **DistilBERT SST-2** ✅ | **91.3%** | ⚡ Fast (~0.1s) | 66M params | Free | ✅ Yes |
| BERT-base-uncased | 93.5% | 🕐 Slower (~0.3s) | 110M params | Free | ✅ Yes |
| RoBERTa-base | 94.8% | 🐢 Slow (~0.4s) | 125M params | Free | ✅ Yes |
| TextBlob (classical) | ~76% | ⚡⚡ Very Fast | Rule-based | Free | ✅ Yes |
| GPT-4o (API) | ~97%+ | ⚡ Fast | Massive | 💰 Paid | ❌ Cloud |
| Llama 3.1 via Groq | ~95%+ | ⚡⚡ Very Fast | 8B params | Free tier | ❌ Cloud |

> **Verdict:** DistilBERT hits the sweet spot — near state-of-the-art accuracy with the smallest footprint for fully local, free inference. The Groq fallback adds cloud-level accuracy only when truly needed.

---

## ✨ Benefits of This System

- **🛡️ Fault-tolerant** — Never crashes silently; errors are caught and routed to the LLM healer
- **💡 Self-aware** — The model knows when it doesn't know, unlike standard classifiers
- **💰 Cost-efficient** — Heavy LLM is only called for hard cases; most predictions are local and free
- **🔒 Privacy-first** — Confident predictions never leave your machine
- **📊 Transparent** — Every decision and confidence score is logged in the state
- **🧩 Extensible** — Add new nodes (e.g., a language detector, topic router) easily in LangGraph
- **🎓 Educational** — Clear code structure, great for learning LangGraph + HuggingFace together

---

## 📦 Dependencies — Exact Versions

> ⚠️ **Use these exact versions** to avoid compatibility issues.  
> Copy and paste the install command below — do not skip version pinning.

```txt
# requirements.txt (pinned versions — tested and working)
langgraph==0.2.74
langchain-core==0.3.29
langchain-groq==0.2.3
transformers==4.47.1
torch==2.5.1
datasets==3.2.0
```

**Python Version Required:**
```
Python 3.10, 3.11, 3.12, or 3.13
```

Check yours:
```bash
python --version
```

---

## 🔑 Getting Your Free Groq API Key

This project uses **Groq** as the fallback LLM provider. Groq offers a **free tier** — no credit card required to get started.

### Step 1 — Sign Up at Groq (Free)

👉 Go to **[https://console.groq.com](https://console.groq.com)**

- Click **"Sign Up"** and create a free account
- Verify your email
- You will land on the Groq console dashboard

### Step 2 — Generate Your API Key

- In the left sidebar, click **"API Keys"**
- Click **"Create API Key"**
- Give it a name (e.g., `self-healing-pipeline`)
- **Copy the key immediately** — it is only shown once!

> 💡 **Free Tier:** Groq's free tier allows generous monthly request limits for models like `llama-3.1-8b-instant`. For this project (which only calls Groq on low-confidence predictions), the free tier is more than enough for learning and testing.

### Step 3 — Paste Your Key into the Project

Open the file `pipeline/langgraph_pipeline.py` and find this section near the top of the file:

```python
# ============================================================
# 🔑 PASTE YOUR GROQ API KEY BELOW
# ─────────────────────────────────────────────────────────────
# How to get a FREE key (no credit card needed):
#   1. Go to https://console.groq.com
#   2. Sign up for a free account
#   3. Click "API Keys" in the left sidebar
#   4. Click "Create API Key" and copy it
# ─────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
#                              ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#                              REPLACE ONLY THIS STRING
#                              Keep the quotes around it
# ============================================================
```

**Example — after replacing:**
```python
os.environ["GROQ_API_KEY"] = "gsk_abc123XYZyourActualKeyGoesHere"
```

> ⚠️ **Security note:** Never share this key publicly or commit it to GitHub. If you accidentally expose it, go back to [https://console.groq.com/keys](https://console.groq.com/keys) and delete + regenerate it immediately.

---

## 🛠️ Step-by-Step Setup Guide

> ⏱️ **Total estimated time: 10–20 minutes** (mostly downloading the DistilBERT model ~260MB)

---

### Step 1 — Get the Project Files

```bash
# If you have the zip file:
unzip SelfHealingPipeline.zip
cd SelfHealingPipeline

# OR if cloning from GitHub:
git clone https://github.com/your-username/SelfHealingPipeline.git
cd SelfHealingPipeline
```

---

### Step 2 — Create a Virtual Environment *(Strongly Recommended)*

A virtual environment keeps your project's packages separate from your system Python.

```bash
# Create the environment (run this once)
python -m venv venv
```

```bash
# Activate it — pick your operating system:

# macOS / Linux:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

> ✅ You should now see `(venv)` at the start of your terminal prompt. This confirms it is active.

---

### Step 3 — Install All Dependencies

```bash
# Install with exact pinned versions (copy-paste this entire command):
pip install langgraph==0.2.74 langchain-core==0.3.29 langchain-groq==0.2.3 transformers==4.47.1 torch==2.5.1 datasets==3.2.0
```

Or if you are using the updated `requirements.txt`:
```bash
pip install -r requirements.txt
```

> ⏱️ This step takes **3–7 minutes**. PyTorch alone is ~700MB.  
> You will see a progress bar for each package — this is normal.

---

### Step 4 — Add Your Groq API Key

Open `pipeline/langgraph_pipeline.py` and replace the placeholder with your actual key:

```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  # ← Replace this string
```

See the [Getting Your Free Groq API Key](#-getting-your-free-groq-api-key) section above for the exact location and instructions.

---

### Step 5 — Download the DistilBERT Model

The model downloads automatically when you first run the pipeline. To download it manually in advance:

```bash
# Run this one-liner from your terminal to pre-download and save the model:
python -c "from transformers import pipeline; p = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english'); p.save_pretrained('./local_model'); print('Done! Model saved to ./local_model')"
```

> ⏱️ First-time download: **2–5 minutes** (~260MB for DistilBERT).  
> After this, it is cached in `./local_model/` and never re-downloaded.

---

## ▶️ How to Run

Make sure your virtual environment is **active** first, then:

### Run All 4 Test Cases

```bash
python examples/test_inputs.py
```

### Run a Single Custom Prediction

```python
# Create a new file or open Python shell:
from pipeline.langgraph_pipeline import pipeline_app

result = pipeline_app.invoke({"text": "This movie was absolutely fantastic!"})

print(f"Prediction:  {result['prediction']}")
print(f"Confidence:  {result['confidence'] * 100:.1f}%")
print(f"Source:      {result['source']}")
```

### Expected Terminal Output

```
============================================================
TEST 1: HIGH CONFIDENCE (>80%)
============================================================
Input Text: 'I absolutely love this product, it is amazing!'

--- FINAL RESULT ---
Prediction: POSITIVE
Confidence: 98.2%
Source:     Local_Model

============================================================
TEST 2: MEDIUM CONFIDENCE (50%-80%)
============================================================
[Station 2A] CLARIFY NODE TRIGGERED! Flagging for human review...

--- FINAL RESULT ---
Prediction: NEEDS CLARIFICATION
Confidence: 65.0%
Source:     Human_Review_Required
Notes:      Confidence was medium (50-80%). Asked user for more context.

============================================================
TEST 3: LOW CONFIDENCE (<50%)
============================================================
[Station 2B] FALLBACK NODE TRIGGERED! Asking the LLM to heal the prediction...

--- FINAL RESULT ---
Prediction: NEGATIVE
Confidence: 100.0%
Source:     Self_Healer_LLM
Notes:      Fixed by LLM Fallback

============================================================
TEST 4: SYSTEM CRASH (Text too long)
============================================================
[Station 2B] FALLBACK NODE TRIGGERED! Asking the LLM to heal the prediction...

--- FINAL RESULT ---
Prediction: POSITIVE
Confidence: 100.0%
Source:     Self_Healer_LLM
Notes:      Fixed by LLM Fallback
```

---

## 📁 Project Structure

```
SelfHealingPipeline/
│
├── pipeline/
│   └── langgraph_pipeline.py    # 🧠 Main pipeline — nodes, router, graph
│
├── examples/
│   └── test_inputs.py           # 🧪 4 test cases covering all routing paths
│
├── model_training/
│   └── fine_tuning_script.py    # 📖 Shows how DistilBERT was fine-tuned (reference)
│
├── evaluation/
│   └── model_metrics.ipynb      # 📊 Accuracy, Precision, Recall, F1 results
│
├── local_model/                 # 💾 Downloaded DistilBERT weights (auto-created)
│   └── (generated on first run)
│
└── requirements.txt             # 📦 Python dependencies
```

---

## ⏱️ Estimated Setup Time

| Step | What Happens | Time |
|---|---|---|
| Unzip / clone project | Extract files | < 1 min |
| Create virtual environment | `python -m venv venv` | < 1 min |
| Install Python packages | PyTorch + LangGraph + Transformers | 3–7 min |
| Download DistilBERT model | ~260MB from HuggingFace | 2–5 min |
| Add Groq API key | Edit one line in the pipeline file | < 1 min |
| **Total (first time)** | | **~8–15 minutes** |
| **Subsequent runs** | Model already cached locally | **~10 seconds to load** |

---

## 🔧 Troubleshooting

**`ModuleNotFoundError: No module named 'langgraph'`**
```bash
# Your virtual environment is probably not active. Run:
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**`AuthenticationError: Invalid Groq API key`**
```bash
# Check pipeline/langgraph_pipeline.py
# Make sure you replaced "your_groq_api_key_here" with your actual key
# Your key should start with: gsk_
# Get a fresh key at: https://console.groq.com/keys
```

**`OSError: local_model directory not found`**
```bash
# Run this to download the model to the correct location:
python -c "from transformers import pipeline; p = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english'); p.save_pretrained('./local_model')"
```

**`pip install` is very slow or stalling on PyTorch**
```bash
# Install the CPU-only version of PyTorch (much smaller, ~200MB vs 700MB):
pip install torch==2.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# Then install the rest:
pip install langgraph==0.2.74 langchain-core==0.3.29 langchain-groq==0.2.3 transformers==4.47.1 datasets==3.2.0
```

**Groq API rate limit hit**
> Groq's free tier has monthly limits. If you hit them, the fallback node will fail. You can check your usage at [https://console.groq.com](https://console.groq.com) and wait for the next billing cycle, or create a new free account.

---

## 📄 License

Open source — free to use, modify, and learn from.

---

*Built with ❤️ using LangGraph · DistilBERT · Groq (Llama 3.1) · HuggingFace Transformers*