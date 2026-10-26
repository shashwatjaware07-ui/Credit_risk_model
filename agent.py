import os
import streamlit as st
from openai import OpenAI

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def generate_underwriting_report(applicant_explanation):
    prob = applicant_explanation["default_probability"]
    decision = applicant_explanation["decision"]
    factors = applicant_explanation["top_factors"]
    
    risk_up = [f"{f['feature']} (impact: +{f['shap']:.3f})" for f in factors if f["direction"] == "increases risk"]
    risk_down = [f"{f['feature']} (impact: {f['shap']:.3f})" for f in factors if f["direction"] == "decreases risk"]
    
    prompt = f"""
    You are an automated, legally compliant banking credit underwriting agent.
    Review the following applicant risk profile:
    - Default Probability: {prob:.2%}
    - Automated Decision: {decision}
    - Key Factors Increasing Risk: {', '.join(risk_up)}
    - Key Factors Reducing Risk: {', '.join(risk_down)}
    
    Provide an executive underwriting report structured as:
    1. Executive Summary: 2 concise sentences explaining the rationale behind the decision.
    2. Primary Adverse Action Reasons: Bulleted list explaining why the risk factors triggered concern.
    3. Counterfactual Approval Path: Actionable guidance on what parameters (e.g., loan amount, down payment) 
       must change to bring this applicant below the 35% institutional default threshold.
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Fallback Local Report:** The API encountered an error ({e}). Decision: {decision} at {prob:.2%} risk."