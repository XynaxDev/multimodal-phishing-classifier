# ml/multimodal_infer_cli.py
#
# Interactive multimodal inference + explanation (text, url, image, combined).
# Put this file in your repo's ml/ folder and run:
#     python ml/multimodal_infer_cli.py
#
# (Part 1/6) — Imports, dotenv, config
import os
import json
import time
import re
from io import BytesIO
from typing import Optional, Dict, Any, Tuple

import dotenv
dotenv.load_dotenv()  # reads .env from project root

import requests
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)

# Optional: HF Router Inference client (chat completions / router)
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except Exception:
    HF_CLIENT_AVAILABLE = False

# ---------------- USER CONFIG / ENV ----------------
# Model paths (adjust if you keep models in different place)
TEXT_MODEL_DIR = os.getenv("MM_TEXT_MODEL_DIR", "backend/models/text/bert_finetuned")
URL_MODEL_DIR  = os.getenv("MM_URL_MODEL_DIR",  "backend/models/url/distilbert_url")
IMAGE_MODEL_PATH = os.getenv("MM_IMAGE_MODEL_PATH", "backend/models/image/best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OCR options
USE_OCR = os.getenv("MM_USE_OCR", "1") == "1"
N8N_OCR_ENDPOINT = os.getenv("MM_N8N_OCR_ENDPOINT", "http://localhost:5678/webhook/ocr")
TESSERACT_CMD = os.getenv("MM_TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
N8N_TIMEOUT = int(os.getenv("MM_N8N_TIMEOUT", "5"))

# Fusion weights (alpha = weight for text in fusion)
DEFAULT_ALPHA = float(os.getenv("MM_FUSION_ALPHA", "0.80"))  # favor text
URL_BINARY_WEIGHT = float(os.getenv("MM_URL_BINARY_WEIGHT", "0.5"))

# LLM / HF Router config
LLM_PROVIDER = os.getenv("MM_LLM_PROVIDER", "hf")  # "hf" or "none" or "gemini"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
MM_HF_MODEL = os.getenv("MM_HF_MODEL", "google/gemma-2-2b-it")  # model served on HF router

# optional Gemini vars (if you use Gemini / AI Studio)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", None)

# ---------------------------------------------------
print("Device:", DEVICE)
print("LLM provider:", LLM_PROVIDER, " HF client avail:", HF_CLIENT_AVAILABLE)

# (Part 2/6) — Transforms, model loading, OCR, and predictor functions

# ---------------- Transforms & helpers ----------------
IMG_SIZE = int(os.getenv("MM_IMG_SIZE", "224"))
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_text_model(text_dir: str):
    print("Loading text model from:", text_dir)
    tokenizer = BertTokenizerFast.from_pretrained(text_dir)
    model = BertForSequenceClassification.from_pretrained(text_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def load_url_model(url_dir: str):
    print("Loading URL model from:", url_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(url_dir)
    model = DistilBertForSequenceClassification.from_pretrained(url_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def build_resnet50_for_loading(state_dict_path: str):
    print("Building ResNet50 and loading weights from:", state_dict_path)
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, 2))
    sd = torch.load(state_dict_path, map_location=DEVICE)
    # handle wrapped dict or DataParallel prefixes
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[nk] = v
        model.load_state_dict(new_sd)
    else:
        # fallback: attempt direct load
        model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------- OCR ----------------
def ocr_n8n(image_bytes: bytes) -> Optional[str]:
    try:
        files = {"file": ("img.png", image_bytes, "image/png")}
        r = requests.post(N8N_OCR_ENDPOINT, files=files, timeout=N8N_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            return j.get("ocr_text") or j.get("text") or j.get("ocr") or None
    except Exception:
        pass
    return None

def ocr_local(image_bytes: bytes) -> Optional[str]:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        txt = pytesseract.image_to_string(img)
        return txt
    except Exception:
        return None

def get_ocr_text_from_path(img_path: str) -> Optional[str]:
    if not USE_OCR:
        return None
    try:
        with open(img_path, "rb") as f:
            b = f.read()
    except Exception:
        return None
    txt = ocr_n8n(b)
    if txt and txt.strip():
        print("[OCR] text from n8n")
        return txt.strip()
    txt = ocr_local(b)
    if txt and txt.strip():
        print("[OCR] text from local Tesseract")
        return txt.strip()
    return None

# ---------------- Predictors ----------------
def predict_text_probs(tokenizer, model, text: Optional[str]) -> Optional[Dict[str, float]]:
    if not text or not str(text).strip():
        return None
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    # ensure shape 2
    if len(probs) >= 2:
        return {"benign": float(probs[0]), "phishing": float(probs[1])}
    return {"benign": float(probs[0]), "phishing": 0.0}

def predict_url_probs(tokenizer, model, url: Optional[str]) -> Optional[Dict[str, float]]:
    if not url or not str(url).strip():
        return None
    enc = tokenizer(url, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    out = {}
    if len(probs) == 3:
        out = {"benign": float(probs[0]), "phishing": float(probs[1]), "malware": float(probs[2])}
    elif len(probs) == 2:
        out = {"benign": float(probs[0]), "phishing": float(probs[1]), "malware": 0.0}
    else:
        # fallback: treat last as phishing
        out = {"benign": float(probs[0]) if len(probs) > 0 else 0.0, "phishing": float(probs[-1]) if len(probs) > 0 else 0.0, "malware": 0.0}
    return out

def predict_image_probs(image_model, img_path: Optional[str]) -> Optional[Dict[str, float]]:
    if not img_path:
        return None
    if not os.path.exists(img_path):
        print("[image] path missing:", img_path)
        return None
    img = Image.open(img_path).convert("RGB")
    x = img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = image_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    if len(probs) >= 2:
        return {"benign": float(probs[0]), "phishing": float(probs[1])}
    return {"benign": float(probs[0]), "phishing": 0.0}


# (Part 3/6) — Fusion helpers and LLM wrappers

# ---------------- Fusion (Option 2: full fusion) ----------------
def url_multiclass_to_binary(url_probs: Dict[str, float], url_weight: float = URL_BINARY_WEIGHT) -> float:
    # Combine phishing and malware probability into single phishing score
    p_phish = url_probs.get("phishing", 0.0)
    p_malware = url_probs.get("malware", 0.0)
    # both count as phishing evidence; simple sum clamped to 1.0
    return float(min(1.0, p_phish + p_malware))

def fuse_all(text_p: Optional[float], url_p_bin: Optional[float], img_p: Optional[float], ocr_p: Optional[float], alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    modalities = {"text": text_p, "url": url_p_bin, "img": img_p, "ocr": ocr_p}
    # choose text_signal = max(text, ocr)
    text_signal = None
    if text_p is not None and ocr_p is not None:
        text_signal = max(text_p, ocr_p)
    elif text_p is not None:
        text_signal = text_p
    elif ocr_p is not None:
        text_signal = ocr_p

    available = {k: v for k, v in [("text", text_signal), ("url", url_p_bin), ("img", img_p)] if v is not None}
    if not available:
        return {"fused_prob": 0.0, "label": 0, **modalities}

    if "text" in available:
        rem = 1.0 - alpha
        others = [k for k in available.keys() if k != "text"]
        if others:
            share = rem / len(others)
            fused = alpha * available["text"]
            for o in others:
                fused += share * available[o]
        else:
            fused = available["text"]
    else:
        fused = sum(available.values()) / len(available)

    label = 1 if fused >= 0.5 else 0
    return {"fused_prob": float(fused), "label": int(label), **modalities}

# ---------------- LLM explanation via HF Router ----------------
def hf_router_explain(prompt: str, hf_api_key: str, model_name: str = None) -> Optional[str]:
    if not hf_api_key or not HF_CLIENT_AVAILABLE:
        return None
    model_name = model_name or MM_HF_MODEL
    try:
        client = InferenceClient(model=model_name, token=hf_api_key)
        resp = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], max_tokens=400, temperature=0.2)
        # attempt multiple extraction patterns
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
            try:
                return choice.message.content
            except Exception:
                return json.dumps(resp, ensure_ascii=False)
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            if "error" in resp:
                return None
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

def make_llm_explanation(summary: Dict[str, Any], ocr_text: Optional[str] = None) -> str:
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

    # prefer hf router
    if LLM_PROVIDER == "hf" and HUGGINGFACE_API_KEY:
        out = hf_router_explain(prompt, HUGGINGFACE_API_KEY)
        if out:
            return out.strip()
    # gemini fallback
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY and GEMINI_ENDPOINT:
        out = explain_with_gemini(prompt, GEMINI_API_KEY, GEMINI_ENDPOINT)
        if out:
            return out.strip()

    # fallback heuristic explanation
    fused = summary.get("fused_prob", 0.0)
    text_p = summary.get("text")
    url_p = summary.get("url")
    img_p = summary.get("img")
    lines = []
    lines.append(f"Final fused phishing probability: {fused:.3f} -> LABEL: {'PHISHING' if summary.get('label')==1 else 'BENIGN'}")
    lines.append("")
    lines.append("Modalities:")
    lines.append(f" - text: {text_p}")
    lines.append(f" - url:  {url_p}")
    lines.append(f" - img:  {img_p}")
    lines.append("")
    if fused >= 0.9:
        lines.append("- Strong evidence of phishing across modalities.")
    elif fused >= 0.6:
        lines.append("- Moderate phishing signals; caution advised.")
    else:
        lines.append("- Likely benign, but verify URLs and sender.")
    if ocr_text:
        lines.append("")
        lines.append("OCR snippet:")
        lines.append(ocr_text.replace("\n", " ")[:400] + ("..." if len(ocr_text)>400 else ""))
    return "\n".join(lines)

# (Part 4/6) — Intent classification and helper functions

# Intent classification using llm
def classify_intent_llm(user_query: str, url: str, img_path: str) -> Optional[str]:
    """
    Use the configured LLM (HF Router or Gemini endpoint) to detect user intent.
    Returns one of: "url_check", "text_message", "image_check", "combined", "chat", "unknown"
    or None if LLM not available (caller can fall back to heuristics).
    """
    if not (user_query or url or img_path):
        return "unknown"
    if not LLM_PROVIDER or LLM_PROVIDER == "none":
        return None

    prompt = f"""
You are an intent classifier for a cybersecurity chatbot.
DO NOT answer the user's question. DO NOT perform security analysis.
Your ONLY job is to classify the user's INTENT.

User query:
"{user_query}"

User-provided URL:
"{url}"

User-provided image path:
"{img_path}"

Choose EXACTLY ONE intent label from:

- chat                     → user is having normal conversation, greetings, asking about you, features, etc.
- url_check                → user wants a URL safety analysis
- text_message             → user pasted an email, SMS, HTML, or message content
- image_check              → user wants you to analyze an image/screenshot
- combined                 → user wants multimodal analysis using multiple inputs
- unknown                  → unclear what user wants

Rules:
1. If the user is chatting (e.g., "how can you help me?", "what can you do?"), choose "chat".
2. If the user writes "check this URL", "is this link safe", or similar → choose "url_check".
3. If the user pastes content (email/SMS/HTML), choose "text_message".
4. If the query mentions "screenshot", "image", "pic", "photo" → choose "image_check".
5. If user provides both URL + text OR text + image OR image + URL → choose "combined".
6. NEVER classify chit-chat as phishing analysis.
7. NEVER output explanation — ONLY return one label.

Return ONLY the label on a single line.
"""
    resp = None
    if LLM_PROVIDER == "hf" and HUGGINGFACE_API_KEY:
        try:
            resp = hf_router_explain(prompt, HUGGINGFACE_API_KEY)
        except Exception:
            resp = None
        if resp:
            txt = str(resp).lower()
            for tok in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                if tok in txt:
                    return tok
            words = (txt.strip().split())
            if words:
                w = words[0].strip().lower()
                if w in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                    return w

    # Gemini fallback
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY and GEMINI_ENDPOINT:
        try:
            resp = explain_with_gemini(prompt, GEMINI_API_KEY, GEMINI_ENDPOINT)
        except Exception:
            resp = None
        if resp:
            txt = str(resp).lower()
            for tok in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                if tok in txt:
                    return tok
            words = (txt.strip().split())
            if words:
                w = words[0].strip().lower()
                if w in ("url_check", "text_message", "image_check", "combined", "chat", "unknown"):
                    return w

    # If LLM calls failed or not available, use a lightweight heuristic fallback:
    q = (user_query or "").lower()
    # ---- HARD OVERRIDE: If user asks about a domain, force URL intent ----
    bare_domain = re.search(r'\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/[^\s]*)?\b', q)
    if bare_domain:
        if any(x in q for x in ["safe", "visit", "open", "check", "legit", "phish", "phishing", "think"]):
            return "url_check"

    if img_path and (("screenshot" in q) or ("image" in q) or ("photo" in q)):
        return "image_check"
    if user_query:
        # rough heuristics for email / sms / script detection
        if "subject:" in q or "dear " in q or "regards" in q or ("http" in q and len(q) > 20):
            return "text_message"
        if "how can you help" in q or q.strip() in ("hi", "hello", "hey", "hii", "hie"):
            return "chat"
        if "url" in q or "link" in q or "visit" in q or "safe" in q or "legit" in q:
            if url:
                if any(word in user_query.lower() for word in ["what", "this", "site", "link", "url", "open", "go"]):
                    return "url_check"

    modalities = sum([1 if user_query else 0, 1 if url else 0, 1 if img_path else 0])
    if modalities >= 2:
        return "combined"
    return "unknown"

# ---------------- INTENT DETECTION (heuristic) ----------------
def heuristic_intent(user_text: str, has_url: bool, has_image: bool) -> str:
    t = user_text.lower() if user_text else ""
    if has_image and ("image" in t or "screenshot" in t or "photo" in t):
        return "image"
    if has_url or "url" in t or "visit" in t or "link" in t or "site" in t:
        return "url"
    if "email" in t:
        return "text_email"
    if "sms" in t:
        return "text_sms"
    if "script" in t or "html" in t or "<html" in t:
        return "text_script"
    return "combined"

# ----------------- Helper functions -----------------
def extract_single_url(text: str) -> Optional[str]:
    if not text:
        return None

    # First remove common trailing punctuation
    cleaned = re.sub(r'[<>\(\)\[\]\{\},;:]+', ' ', text)

    # full URL match
    full = re.search(r'(https?://[^\s]+)', cleaned)
    if full:
        return full.group(1).rstrip(".,;:!?)(")

    # bare domain match
    dom = re.search(r'\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/[^\s]*)?\b', cleaned)
    if dom:
        url = dom.group(0).rstrip(".,;:!?)(")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    return None


def friendly_chat_reply_llm(user_query: str) -> str:
    sec_keywords = ("url", "link", "safe", "phish", "phishing", "email", "screenshot", "login", "credentials", "visit")
    if not any(k in (user_query or "").lower() for k in sec_keywords):
        return "I can help with phishing and URL/image/text safety checks — paste a URL, screenshot, or the message you want me to check."
    prompt = f"""
You are CyberGuardian — a friendly cybersecurity assistant.
User asked: "{user_query}"

Do NOT perform any classification here. Provide a short, friendly answer (2-4 sentences) giving general safety advice and how the user can ask for a concrete check (URL/text/image).
Be concise and avoid hallucination.
    """
    if LLM_PROVIDER == "hf" and HUGGINGFACE_API_KEY:
        out = hf_router_explain(prompt, HUGGINGFACE_API_KEY)
        if out:
            return out.strip()
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY and GEMINI_ENDPOINT:
        out = explain_with_gemini(prompt, GEMINI_API_KEY, GEMINI_ENDPOINT)
        if out:
            return out.strip()
    return "I can check URLs, emails, SMS and screenshots for phishing indicators. Paste the URL, message, or image path and I'll run the relevant checks."

# (Part 5/6) — interactive_main: text, url, image, combined branches

# ---------------- Interactive CLI ----------------
def interactive_main():
    print("\n======== Multimodal Inference CLI ========")
    print("1. Text Classification")
    print("2. URL Safety Check")
    print("3. Image Classification")
    print("4. Combined (manual: text + URL + image)")
    print("5. Chat Mode (Smart intent detection)")
    print("==========================================\n")

    choice = input("Enter choice (1–5): ").strip()

    # ---------------------------------------------------------------------
    # 1) TEXT ONLY INFERENCE
    # ---------------------------------------------------------------------
    if choice == "1":
        text = input("Enter text/email/SMS: ").strip()
        tokenizer, model = load_text_model(TEXT_MODEL_DIR)
        probs = predict_text_probs(tokenizer, model, text)

        print("\n===== TEXT RESULT =====")
        print(probs)
        label = "PHISHING" if probs and probs.get("phishing", 0.0) > probs.get("benign", 0.0) else "BENIGN"
        print("LABEL:", label)

        explanation = make_llm_explanation({"mode": "text", "text": text, "probs": probs, "label": label}, None)
        print("\n" + (explanation or "(No explanation available)"))
        return

    # ---------------------------------------------------------------------
    # 2) URL ONLY
    # ---------------------------------------------------------------------
    if choice == "2":
        url = input("Enter URL: ").strip()
        tokenizer, model = load_url_model(URL_MODEL_DIR)
        probs = predict_url_probs(tokenizer, model, url)

        print("\n===== URL RESULT =====")
        print(probs)
        label = max(probs, key=lambda x: probs[x]) if probs else "unknown"
        print("LABEL:", label.upper() if isinstance(label, str) else label)

        explanation = make_llm_explanation({"mode": "url", "url": url, "probs": probs, "label": label}, None)
        print("\n" + (explanation or "(No explanation available)"))
        return

    # ---------------------------------------------------------------------
    # 3) IMAGE ONLY
    # ---------------------------------------------------------------------
    if choice == "3":
        img_path = input("Enter image path: ").strip()
        image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
        probs = predict_image_probs(image_model, img_path)

        ocr_text = get_ocr_text_from_path(img_path) if USE_OCR else None
        label = "PHISHING" if probs and probs.get("phishing", 0.0) > probs.get("benign", 0.0) else "BENIGN"

        print("\n===== IMAGE RESULT =====")
        print(probs)
        print("LABEL:", label)
        if ocr_text:
            print("\nOCR snippet:", ocr_text[:200])

        explanation = make_llm_explanation({"mode": "image", "probs": probs, "ocr": ocr_text, "label": label}, ocr_text)
        print("\n" + (explanation or "(No explanation available)"))
        return

    # ---------------------------------------------------------------------
    # 4) COMBINED MODE (manual: text + URL + image)
    # ---------------------------------------------------------------------
    if choice == "4":
        text = input("Enter text (optional): ").strip() or None
        url = input("Enter URL (optional): ").strip() or None
        img_path = input("Enter image path (optional): ").strip() or None

        # Load models as needed
        tokenizer_text = tokenizer_url = None
        text_model = url_model = image_model = None

        if text:
            tokenizer_text, text_model = load_text_model(TEXT_MODEL_DIR)
        if url:
            tokenizer_url, url_model = load_url_model(URL_MODEL_DIR)
        if img_path:
            image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)

        # Get probs
        text_probs = predict_text_probs(tokenizer_text, text_model, text) if text else None
        url_probs = predict_url_probs(tokenizer_url, url_model, url) if url else None
        img_probs = predict_image_probs(image_model, img_path) if img_path else None

        ocr_text = get_ocr_text_from_path(img_path) if img_path else None
        ocr_prob = None
        if ocr_text and tokenizer_text:
            tmp = predict_text_probs(tokenizer_text, text_model, ocr_text)
            ocr_prob = tmp["phishing"] if tmp else None

        # fusion numeric inputs
        text_p = text_probs["phishing"] if text_probs else None
        url_p = url_multiclass_to_binary(url_probs) if url_probs else None
        img_p = img_probs["phishing"] if img_probs else None

        fusion = fuse_all(text_p, url_p, img_p, ocr_prob, alpha=DEFAULT_ALPHA)
        fusion.update({"raw_text": text_probs, "raw_url": url_probs, "raw_img": img_probs})

        print("\n===== COMBINED RESULT =====")
        print("Fused probability:", fusion["fused_prob"])
        print("Label:", "PHISHING" if fusion["label"] == 1 else "BENIGN")

        explanation = make_llm_explanation(fusion, ocr_text)
        print("\n" + (explanation or "(No explanation available)"))
        return

# (Part 6/6) — Chat mode + combined routing + main

    # ---------------------------------------------------------------------
    # 5) CHAT MODE — SMART INTENT DETECTION (NEW FEATURE)
    # ---------------------------------------------------------------------
    if choice == "5":
        print("\n====== Chat Mode ======")
        print("Talk naturally. Paste text, URLs, or mention image paths.")
        print("Example: 'is https://xyz-login.com safe?'")
        print("======================================\n")

        user_query = input("You: ").strip()

        # Extract URL and detect first
        extracted_url = extract_single_url(user_query)
        url = extracted_url

        # If user included an image path in the query (simple heuristic)
        img_path_candidate = None
        m = re.search(r'([A-Za-z]:(?:\\|/)[^\s]+(?:\.png|\.jpg|\.jpeg))', user_query)
        if m:
            img_path_candidate = m.group(1)

        # LLM intent detection (try LLM; fallback to heuristics)
        # LLM intent detection
        intent = classify_intent_llm(user_query, url, None)
        print(f"[INFO] intent detected → {intent}")

        # -----------------------------------------------------
        # HARD OVERRIDE: if any URL/domain is extracted → URL CHECK
        # -----------------------------------------------------
        if url:
            print("[INFO] override → domain detected in text, forcing url_check")
            intent = "url_check"

        if intent is None:
            # LLM unavailable -> fallback to heuristic
            hint = heuristic_intent(user_query, bool(url), bool(img_path_candidate))
            if hint.startswith("text"):
                intent = "text_message"
            elif hint == "url":
                intent = "url_check"
            elif hint == "image":
                intent = "image_check"
            elif hint == "combined":
                intent = "combined"
            else:
                intent = "unknown"

        print(f"[INFO] intent detected → {intent}")

        # CASE: chat or unknown -> friendly reply or ask for clarifying info
        if intent in ("chat", "unknown"):
            reply = friendly_chat_reply_llm(user_query)
            print("\nCyberGuardian:\n", reply)
            return

        # CASE: url_check -> run URL model only (user expects url-only)
        if intent == "url_check":
            url_to_check = url or input("Enter URL to check: ").strip()
            tokenizer_url, url_model = load_url_model(URL_MODEL_DIR)
            url_probs = predict_url_probs(tokenizer_url, url_model, url_to_check)

            label = max(url_probs, key=url_probs.get) if url_probs else "unknown"
            summary = {"url": url_probs, "label": label}

            print("\n===== URL SAFETY RESULT =====")
            print(url_probs)
            print("LABEL:", label.upper() if isinstance(label, str) else label)

            explanation = make_llm_explanation(summary, None)
            print("\nCyberGuardian:\n", explanation)
            return

        # CASE: text_message -> run text model only
        if intent == "text_message":
            tokenizer_text, text_model = load_text_model(TEXT_MODEL_DIR)
            text_probs = predict_text_probs(tokenizer_text, text_model, user_query)

            label = "phishing" if text_probs and text_probs.get("phishing",0.0) > text_probs.get("benign",0.0) else "benign"
            summary = {"text": text_probs, "label": label}

            print("\n===== TEXT CLASSIFICATION =====")
            print(text_probs)
            print("LABEL:", label.upper())
            explanation = make_llm_explanation(summary, None)
            print("\nCyberGuardian:\n", explanation)
            return

        # CASE: image_check -> run image model + OCR (if provided)
        if intent == "image_check":
            img_path = img_path_candidate or input("Enter image path to analyze: ").strip()
            image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
            img_probs = predict_image_probs(image_model, img_path)
            ocr_text = get_ocr_text_from_path(img_path) if USE_OCR else None
            label = "PHISHING" if img_probs and img_probs.get("phishing",0.0) > img_probs.get("benign",0.0) else "BENIGN"

            print("\n===== IMAGE CHECK RESULT =====")
            print(img_probs)
            print("LABEL:", label)
            if ocr_text:
                print("\nOCR snippet:", ocr_text[:300])

            explanation = make_llm_explanation({"img": img_probs, "label": label}, ocr_text)
            print("\nCyberGuardian:\n", explanation)
            return

        # CASE: combined (text + url or text+image or url+image) -> fused analysis
        if intent == "combined":
            # extract url if present
            url_in_query = url or extract_single_url(user_query)
            # load models
            tokenizer_text, text_model = load_text_model(TEXT_MODEL_DIR)
            tokenizer_url, url_model = load_url_model(URL_MODEL_DIR)

            text_probs = predict_text_probs(tokenizer_text, text_model, user_query)
            url_probs = predict_url_probs(tokenizer_url, url_model, url_in_query) if url_in_query else None

            # OCR if image included
            img_p = None
            ocr_text = None
            if img_path_candidate:
                image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
                img_probs = predict_image_probs(image_model, img_path_candidate)
                img_p = img_probs["phishing"] if img_probs else None
                ocr_text = get_ocr_text_from_path(img_path_candidate) if USE_OCR else None
            else:
                img_probs = None

            summary = fuse_all(
                text_probs["phishing"] if text_probs else None,
                url_multiclass_to_binary(url_probs) if url_probs else None,
                img_p,
                None,
                alpha=DEFAULT_ALPHA
            )
            summary["raw_text"] = text_probs
            summary["raw_url"] = url_probs
            summary["raw_img"] = img_probs if img_path_candidate else None

            print("\n===== MULTIMODAL FUSION RESULT =====")
            print(summary)

            explanation = make_llm_explanation(summary, ocr_text)
            print("\nCyberGuardian:\n", explanation)
            return

    print("Invalid option. Try again.")

if __name__ == "__main__":
    interactive_main()
