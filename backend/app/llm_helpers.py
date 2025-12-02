# backend/app/llm_helpers.py
import os
from dotenv import load_dotenv

# Load .env from the project root (2 levels up from this file)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

import json
import requests
from typing import Optional
from .predictors import extract_single_url  # (if needed)
# HF Router client optional
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except Exception:
    HF_CLIENT_AVAILABLE = False



LLM_PROVIDER = os.getenv("MM_LLM_PROVIDER","gemini")  # "hf", "gemini", or "none"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
MM_HF_MODEL = os.getenv("MM_HF_MODEL", "google/gemma-2-2b-it")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", None)

def hf_router_explain(prompt: str, hf_api_key: str, model_name: str = None) -> Optional[str]:
    if not hf_api_key or not HF_CLIENT_AVAILABLE:
        return None
    model_name = model_name or MM_HF_MODEL
    try:
        client = InferenceClient(model=model_name, token=hf_api_key)
        resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], max_tokens=400, temperature=0.2)
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
            try:
                return choice.message.content
            except Exception:
                return json.dumps(resp, ensure_ascii=False)
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            return json.dumps(resp, ensure_ascii=False)
        if isinstance(resp, str):
            return resp
    except Exception as e:
        print("[LLM] HF Router call failed:", e)
    return None

def explain_with_gemini(prompt: str, api_key: str, endpoint: Optional[str]) -> Optional[str]:
    if not api_key or not endpoint:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt}
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            j = r.json()
            for k in ("output", "text", "response", "content", "candidates"):
                if k in j:
                    val = j[k]
                    if isinstance(val, list) and len(val) > 0:
                        return val[0] if isinstance(val[0], str) else json.dumps(val[0])
                    if isinstance(val, str):
                        return val
            return json.dumps(j)
    except Exception as e:
        print("[LLM] Gemini call failed:", e)
    return None

def make_llm_explanation(summary: dict, ocr_text: Optional[str] = None) -> str:
    prompt = f"""
You are CyberGuardian — a friendly cybersecurity assistant.
Your task is to explain the MODEL'S detection result.

IMPORTANT CONSTRAINTS:
- NEVER invent information
- NEVER assume what the image or text contains unless OCR/text is provided
- NEVER fabricate URLs, senders, brands, or login pages
- ONLY use the numeric probabilities supplied
- If probabilities are close, mention uncertainty
- Tone: clear, calm, helpful, slightly conversational (NOT robotic)
- Provide short 1–2 sentence summary + 3–5 helpful bullets

Detection Summary (model outputs):
{json.dumps(summary, indent=2)}

OCR text (if available):
"{(ocr_text[:300] + '...') if ocr_text and len(ocr_text)>300 else (ocr_text or 'None')}"

Write a friendly explanation:
1. Open with a short summary sentence.
2. Explain which signals pushed the model toward phishing/benign.
3. Mention if the evidence is strong/weak/mixed.
4. Give simple, practical safety advice.
5. Keep the explanation concise and human — no overly formal tone.

Return only the explanation, no extra commentary.
    """
    # try HF Router
    if LLM_PROVIDER == "hf" and HUGGINGFACE_API_KEY and HF_CLIENT_AVAILABLE:
        out = hf_router_explain(prompt, HUGGINGFACE_API_KEY)
        if out:
            return out.strip()
    # try Gemini
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY and GEMINI_ENDPOINT:
        out = explain_with_gemini(prompt, GEMINI_API_KEY, GEMINI_ENDPOINT)
        if out:
            return out.strip()
    # fallback deterministic explanation (safe)
    fused = summary.get("fused_prob", None)
    lines = []
    if fused is not None:
        lines.append(f"Final fused phishing probability: {fused:.3f} -> LABEL: {'PHISHING' if summary.get('label')==1 else 'BENIGN'}")
    else:
        # per-mode
        if "probs" in summary:
            probs = summary["probs"]
            if isinstance(probs, dict):
                top = max(probs, key=probs.get)
                lines.append(f"Model verdict: {top.upper()} (phishing: {probs.get('phishing')}, benign: {probs.get('benign')})")
    lines.append("")
    if summary.get("text") is not None:
        lines.append(f"Text signal: {summary.get('text')}")
    if summary.get("url") is not None:
        lines.append(f"URL signal: {summary.get('url')}")
    if summary.get("img") is not None:
        lines.append(f"Image signal: {summary.get('img')}")
    lines.append("")
    lines.append("Advice:")
    if fused is not None:
        if fused >= 0.9:
            lines.append("- Strong evidence of phishing. Do not click links or enter credentials.")
        elif fused >= 0.6:
            lines.append("- Moderate signals. Verify sender and avoid entering sensitive info.")
        else:
            lines.append("- Likely benign but double-check suspicious links and senders.")
    else:
        lines.append("- When in doubt, do not click links or provide credentials.")
    if ocr_text:
        lines.append("")
        lines.append("OCR snippet:")
        lines.append(ocr_text.replace("\n", " ")[:400] + ("..." if len(ocr_text)>400 else ""))
    return "\n".join(lines)

def friendly_chat_reply_llm(user_query: str) -> str:
    prompt = f"""
You are CyberGuardian — a friendly cybersecurity assistant.
User asked: "{user_query}"

Provide a helpful, conversational response (2-4 sentences) about cybersecurity safety and how the user can use this tool for phishing detection.
Be concise, friendly, and avoid hallucination.
If they ask general questions, explain what you can help with (URL checking, email analysis, image screening, etc.).
    """
    # prefer HF
    if LLM_PROVIDER == "hf" and HUGGINGFACE_API_KEY and HF_CLIENT_AVAILABLE:
        out = hf_router_explain(prompt, HUGGINGFACE_API_KEY)
        if out:
            return out.strip()
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY and GEMINI_ENDPOINT:
        out = explain_with_gemini(prompt, GEMINI_API_KEY, GEMINI_ENDPOINT)
        if out:
            return out.strip()
    return "I can check URLs, emails, SMS and screenshots for phishing indicators. Paste the URL, message, or image path and I'll run the relevant checks."
