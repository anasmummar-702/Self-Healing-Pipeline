# Self-Healing Text Classification Pipeline

## 1. Project Overview
This project demonstrates a robust, self-healing text classification pipeline. It integrates a local fine-tuned Transformer model (DistilBERT) with a LangGraph Directed Acyclic Graph (DAG) workflow. Instead of blindly trusting the local model, the system evaluates prediction confidence. If confidence is low or an error occurs, it intelligently routes the data to a "Self-Healer" Large Language Model (LLM) for a fallback prediction.

## 2. Architecture Diagram
```text
      [ User Input Text ]
              ↓
  ( Node 1: Local Predictor ) <-- DistilBERT Model
              ↓
    [ Confidence Router ]
    /         |         \
 >0.80   0.50 - 0.80   <0.50 (or Error)
  /           |           \
ACCEPT     CLARIFY     FALLBACK
  |           |           |
[END][END]   ( Node 2: Self-Healer LLM )
                          |
                        [END]