from flask import Flask, request, jsonify
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

# ---- LOAD ENV ----
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---- OPENAI CLIENT ----
client = OpenAI(api_key=api_key)

# ---- STATE ----
class AgentState(TypedDict):
    input_data: str
    analysis: str
    risks: str
    strategy: str
    final_output: Dict

# ---- HELPER FUNCTION ----
def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior business AI analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# ---- AGENTS ----

# 1. DATA ANALYST
def data_analyst_agent(state: AgentState):
    prompt = f"""
    Analyze the following business data:

    {state['input_data']}

    Focus on:
    - Regional performance
    - Product performance
    - Trends

    Give structured observations.
    """
    analysis = call_llm(prompt)
    return {"analysis": analysis}


# 2. RISK AGENT
def risk_agent(state: AgentState):
    prompt = f"""
    Based on this analysis:

    {state['analysis']}

    Identify:
    - Business risks
    - Market risks
    - Operational risks
    """
    risks = call_llm(prompt)
    return {"risks": risks}


# 3. STRATEGY AGENT
def strategy_agent(state: AgentState):
    prompt = f"""
    Based on:
    Analysis: {state['analysis']}
    Risks: {state['risks']}

    Provide:
    - Strategic recommendations
    - Growth opportunities
    - Cost optimizations
    """
    strategy = call_llm(prompt)
    return {"strategy": strategy}


# 4. SUMMARY AGENT
def summary_agent(state: AgentState):
    prompt = f"""
    Convert the following into STRICT JSON:

    Analysis:
    {state['analysis']}

    Risks:
    {state['risks']}

    Strategy:
    {state['strategy']}

    Format:
    {{
      "insights": [],
      "risks": [],
      "recommendations": []
    }}

    Only return valid JSON.
    """
    output = call_llm(prompt)

    try:
        parsed = json.loads(output)
    except:
        parsed = {"raw_output": output}

    return {"final_output": parsed}


# ---- LANGGRAPH ----
builder = StateGraph(AgentState)

builder.add_node("analyst_node", data_analyst_agent)
builder.add_node("risk_node", risk_agent)
builder.add_node("strategy_node", strategy_agent)
builder.add_node("summary_node", summary_agent)

builder.set_entry_point("analyst_node")
builder.add_edge("analyst_node", "risk_node")
builder.add_edge("risk_node", "strategy_node")
builder.add_edge("strategy_node", "summary_node")
builder.add_edge("summary_node", END)

graph = builder.compile()

# ---- FLASK APP ----
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json.get("data")

    result = graph.invoke({
        "input_data": json.dumps(data),
        "analysis": "",
        "risks": "",
        "strategy": "",
        "final_output": {}
    })

    return jsonify(result["final_output"])


# ---- RUN ----
if __name__ == "__main__":
    app.run(debug=True)