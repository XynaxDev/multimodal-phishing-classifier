# backend/app/predictors.py
import os
import re
import json
from io import BytesIO
from typing import Optional, Dict
from PIL import Image
import requests
import torch
from torchvision import transforms
from .models_loader import DEVICE

IMG_SIZE = int(os.getenv("MM_IMG_SIZE", "224"))
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# OCR configuration
USE_OCR = os.getenv("MM_USE_OCR", "1") == "1"
N8N_OCR_ENDPOINT = os.getenv("MM_N8N_OCR_ENDPOINT", "http://localhost:5678/webhook/ocr")
TESSERACT_CMD = os.getenv("MM_TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
N8N_TIMEOUT = int(os.getenv("MM_N8N_TIMEOUT", "5"))

def predict_text_probs(tokenizer, model, text: Optional[str]) -> Optional[Dict[str, float]]:
    if not text or not text.strip():
        return None
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
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
    if len(probs) == 3:
        return {"benign": float(probs[0]), "phishing": float(probs[1]), "malware": float(probs[2])}
    if len(probs) == 2:
        return {"benign": float(probs[0]), "phishing": float(probs[1]), "malware": 0.0}
    return {"benign": float(probs[0]) if probs else 0.0, "phishing": float(probs[-1]) if probs else 0.0, "malware": 0.0}

def predict_image_probs(image_model, img_path: Optional[str]) -> Optional[Dict[str, float]]:
    if not img_path or not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    x = img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = image_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    if len(probs) >= 2:
        return {"benign": float(probs[0]), "phishing": float(probs[1])}
    return {"benign": float(probs[0]), "phishing": 0.0}

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
        return txt.strip()
    txt = ocr_local(b)
    if txt and txt.strip():
        return txt.strip()
    return None

# improved domain extraction
FULL_URL_PATTERN = re.compile(r'(https?://[^\s)>\]]+)', flags=re.I)
BARE_DOMAIN_PATTERN = re.compile(r'(?<![@/\w.-])([a-zA-Z0-9-]{2,}\.[a-zA-Z]{2,})(/[^\s)>]*)?', flags=re.I)

def extract_single_url(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip()
    m = FULL_URL_PATTERN.search(cleaned)
    if m:
        url = m.group(1).rstrip(".,;:!?)(")
        return url
    m2 = BARE_DOMAIN_PATTERN.search(cleaned)
    if m2:
        candidate = m2.group(0).strip().rstrip(".,;:!?)(")
        if re.search(r'\.[a-zA-Z]{2,}$', candidate):
            if not re.search(r'\b[\w\.-]+@' + re.escape(candidate) + r'\b', cleaned):
                if not candidate.startswith(("http://", "https://")):
                    candidate = "https://" + candidate
                return candidate
    return None

def url_multiclass_to_binary(url_probs: Dict[str, float]) -> float:
    p_phish = url_probs.get("phishing", 0.0)
    p_malware = url_probs.get("malware", 0.0)
    return float(min(1.0, p_phish + p_malware))

def fuse_all(text_p, url_p_bin, img_p, ocr_p, alpha=0.8):
    modalities = {"text": text_p, "url": url_p_bin, "img": img_p, "ocr": ocr_p}
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
