import os
import operator
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- SETUP: Set your FREE Groq API Key ---
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# ==========================================================
# 1. THE STATE (The Notepad passed between agents)
# ==========================================================
class DebateState(TypedDict):
    topic: str
    # 'Annotated[list, operator.add]' means every time an agent speaks, 
    # it APPENDS their message to the history, rather than erasing it!
    history: Annotated[list, operator.add] 
    current_round: int
    max_rounds: int

# ==========================================================
# 2. THE MODELS (The AI Personalities)
# ==========================================================
# We use Meta's Llama 3 model hosted on Groq for FREE!
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# ==========================================================
# 3. THE NODES (The Debaters and the Judge)
# ==========================================================

def pro_agent_node(state: DebateState):
    """Agent 1: Argues IN FAVOR of the topic."""
    print(f"\n--- [PRO AGENT] Thinking... ---")
    
    topic = state["topic"]
    history_text = "\n".join(state["history"])
    
    prompt = (
        f"Topic: {topic}\n"
        f"You are the PRO debater. You strongly AGREE with the topic.\n"
        f"Here is the debate history so far:\n{history_text}\n"
        f"Write a short, fierce 2-sentence argument defending your side or attacking the CON side."
    )
    
    response = llm.invoke(prompt)
    return {"history":[f"PRO: {response.content}"]}

def con_agent_node(state: DebateState):
    """Agent 2: Argues AGAINST the topic."""
    print(f"---[CON AGENT] Thinking... ---")
    
    topic = state["topic"]
    history_text = "\n".join(state["history"])
    
    prompt = (
        f"Topic: {topic}\n"
        f"You are the CON debater. You strongly DISAGREE with the topic.\n"
        f"Here is the debate history so far:\n{history_text}\n"
        f"Write a short, fierce 2-sentence counter-argument attacking the PRO side."
    )
    
    response = llm.invoke(prompt)
    
    # We append the message AND increase the round counter by 1
    return {
        "history": [f"CON: {response.content}"], 
        "current_round": state["current_round"] + 1
    }

def judge_node(state: DebateState):
    """The Judge: Summarizes the debate and declares a winner."""
    print(f"\n--- [JUDGE] Summarizing the debate... ---")
    
    history_text = "\n".join(state["history"])
    
    prompt = (
        f"You are an impartial Judge. Read the following debate:\n{history_text}\n\n"
        f"Provide a brief 3-sentence summary of the arguments, and declare who won (PRO or CON) and why."
    )
    
    response = llm.invoke(prompt)
    return {"history":[f"\n=== FINAL VERDICT ===\n{response.content}"]}

# ==========================================================
# 4. THE ROUTER (The Moderator)
# ==========================================================

def debate_router(state: DebateState):
    """Decides whether to keep debating or go to the judge."""
    if state["current_round"] >= state["max_rounds"]:
        print(f"    [Moderator] Max rounds reached! Sending to Judge.")
        return "go_to_judge"
    else:
        print(f"    [Moderator] Round {state['current_round']} finished. Back to PRO.")
        return "next_round"

# ==========================================================
# 5. BUILD THE LANGGRAPH
# ==========================================================
workflow = StateGraph(DebateState)

workflow.add_node("Pro_Agent", pro_agent_node)
workflow.add_node("Con_Agent", con_agent_node)
workflow.add_node("Judge", judge_node)

workflow.set_entry_point("Pro_Agent")
workflow.add_edge("Pro_Agent", "Con_Agent")

workflow.add_conditional_edges(
    "Con_Agent",
    debate_router,
    {
        "next_round": "Pro_Agent",   
        "go_to_judge": "Judge"       
    }
)

workflow.add_edge("Judge", END)
debate_app = workflow.compile()

# ==========================================================
# 6. RUN THE DEBATE
# ==========================================================
if __name__ == "__main__":
    
    initial_state = {
        "topic": "Artificial Intelligence should be allowed to grade university exams.",
        "history":[],
        "current_round": 0,
        "max_rounds": 2 
    }
    
    print("========================================")
    print(f"DEBATE TOPIC: {initial_state['topic']}")
    print("========================================")
    
    final_state = debate_app.invoke(initial_state)
    
    print("\n\n" + "="*40)
    print("FULL DEBATE TRANSCRIPT:")
    print("="*40)
    for message in final_state["history"]:
        print(message + "\n")