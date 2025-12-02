# backend/app/intent.py
import os
import re
from typing import Optional
from .predictors import extract_single_url
from .llm_helpers import hf_router_explain, explain_with_gemini, LLM_PROVIDER, HUGGINGFACE_API_KEY, GEMINI_API_KEY, GEMINI_ENDPOINT

FORCE_URL_OVERRIDE = os.getenv("MM_FORCE_URL_OVERRIDE", "0") == "1"
SHORT_URL_THRESHOLD = int(os.getenv("MM_SHORT_URL_THRESHOLD", "120"))

def _looks_like_email_text(q: str) -> bool:
    ql = q.lower()
    email_indicators = ["subject:", "dear ", "regards", "click here", "activate", "account", "mailbox", "we will", "please login", "verify", "urgent", "click", "sincerely"]
    if any(ind in ql for ind in email_indicators):
        return True
    if "http" in ql and len(ql) > 200:
        return True
    return False

def heuristic_intent(user_query: str, provided_image_path: Optional[str] = None) -> str:
    q = (user_query or "").strip()
    if not q and not provided_image_path:
        return "unknown"
    if provided_image_path:
        return "image_check"
    extracted = extract_single_url(q)
    has_url = bool(extracted)
    if has_url:
        if FORCE_URL_OVERRIDE:
            return "url_check"
        if _looks_like_email_text(q):
            return "text_message"
        if len(q) > SHORT_URL_THRESHOLD:
            return "text_message"
        ql = q.lower()
        url_words = ["is this site", "is this link", "is this url", "is this website", "link safe", "site safe", "check this link", "check url", "is https", "is http"]
        if any(w in ql for w in url_words):
            return "url_check"
        tokens = q.split()
        url_like_tokens = sum(1 for t in tokens if '.' in t or t.startswith("http"))
        if len(tokens) <= 3 and url_like_tokens >= 1:
            return "url_check"
        return "text_message"
    if _looks_like_email_text(q):
        return "text_message"
    if q.lower() in ("hi", "hello", "hey", "how are you", "how are you?"):
        return "chat"
    if any(w in q.lower() for w in ("screenshot", "image", "photo", "attachment")):
        return "image_check"
    return "chat"

def classify_intent_llm(user_query: str, provided_image_path: Optional[str] = None) -> Optional[str]:
    """
    Try LLM-based intent first (if configured), then fallback to heuristic_intent.
    Returns one of: url_check, text_message, image_check, chat, combined, unknown
    """
    q = user_query or ""
    if LLM_PROVIDER == "hf" and HUGGINGFACE_API_KEY:
        prompt = f"""
You are an intent classifier for a cybersecurity chatbot.
DO NOT answer the user's question. DO NOT perform security analysis.
Your ONLY job is to classify the user's INTENT.

User query:
"{q}"

User-provided image path:
"{provided_image_path}"

Choose EXACTLY ONE intent label from:
- chat
- url_check
- text_message
- image_check
- combined
- unknown

Follow rules:
1. If the user is chatting choose "chat".
2. If the user writes "check this URL", choose "url_check".
3. If the user pasted an email or SMS, choose "text_message".
4. If the user mentions screenshot/image, choose "image_check".
Return ONLY the label.
"""
        try:
            out = hf_router_explain(prompt, HUGGINGFACE_API_KEY)
            if out:
                txt = str(out).lower()
                for tok in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                    if tok in txt:
                        return tok
                w = txt.strip().split()[0] if txt.strip() else None
                if w in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                    return w
        except Exception:
            pass

    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY and GEMINI_ENDPOINT:
        prompt = f"INTENT CLASSIFIER: {q}\nReturn one of: url_check, text_message, image_check, chat, combined, unknown"
        try:
            out = explain_with_gemini(prompt, GEMINI_API_KEY, GEMINI_ENDPOINT)
            if out:
                txt = str(out).lower()
                for tok in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                    if tok in txt:
                        return tok
                w = txt.strip().split()[0] if txt.strip() else None
                if w in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                    return w
        except Exception:
            pass

    # fallback
    return heuristic_intent(user_query, provided_image_path)
