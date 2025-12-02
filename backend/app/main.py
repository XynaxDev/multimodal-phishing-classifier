# backend/app/main.py
import os
from dotenv import load_dotenv

# Define paths and load .env FIRST
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

print(f"Loading .env from: {ENV_PATH}")
print(f"Root directory: {ROOT_DIR}")
print(f".env exists: {os.path.exists(ENV_PATH)}")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print("✅ .env loaded successfully")
else:
    print("❌ .env file not found, using environment variables")


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .models_loader import load_text_model, load_url_model, build_resnet50_for_loading, get_cache_keys, DEVICE
from .predictors import (
    predict_text_probs, predict_url_probs, predict_image_probs,
    get_ocr_text_from_path, extract_single_url, url_multiclass_to_binary, fuse_all
)
from .intent import classify_intent_llm, heuristic_intent
from .llm_helpers import make_llm_explanation, friendly_chat_reply_llm, LLM_PROVIDER

print(f"Loading .env from: {ENV_PATH}")
print(f"Root directory: {ROOT_DIR}")
print(f".env exists: {os.path.exists(ENV_PATH)}")



TEXT_MODEL_DIR = os.getenv("MM_TEXT_MODEL_DIR", "backend/models/text/bert_finetuned")
URL_MODEL_DIR = os.getenv("MM_URL_MODEL_DIR", "backend/models/url/distilbert_url")
IMAGE_MODEL_PATH = os.getenv("MM_IMAGE_MODEL_PATH", "backend/models/image/best_model.pth")
USE_LLM_INTENT = os.getenv("MM_USE_LLM_INTENT", "0") == "1"

print(f"Text model dir: {TEXT_MODEL_DIR}")
print(f"URL model dir: {URL_MODEL_DIR}")
print(f"Image model path: {IMAGE_MODEL_PATH}")
print(f"Use LLM intent: {USE_LLM_INTENT}")

app = FastAPI(title="ph5 Multimodal API")

@app.on_event("startup")
async def startup_event():
    print("🚀 ph5 Multimodal API is starting up...")
    print(f"📡 API will be available at: http://localhost:8000")
    print(f"📖 API docs at: http://localhost:8000/docs")
    print(f"🔧 Device: {DEVICE}")
    print(f"🤖 LLM Provider: {LLM_PROVIDER}")
    print("✅ Startup complete!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly allow methods
    allow_headers=["*"],  # Allows all headers
)

class TextCheckReq(BaseModel):
    text: str

class URLCheckReq(BaseModel):
    url: str

class ChatReq(BaseModel):
    query: str
    image_path: Optional[str] = None

@app.get("/api/status")
def status():
    return {"ok": True, "models_cached": get_cache_keys(), "llm_provider": LLM_PROVIDER, "use_llm_intent": USE_LLM_INTENT}

@app.post("/api/text_check")
def text_check(req: TextCheckReq):
    try:
        tokenizer, model = load_text_model(TEXT_MODEL_DIR)
        probs = predict_text_probs(tokenizer, model, req.text)
        
        # Handle case where prediction fails
        if not probs:
            return {"error": "Failed to get prediction. Model may not be loaded properly."}
            
        # Use consistent verdict logic: highest probability wins
        label = "PHISHING" if probs.get("phishing",0.0) > probs.get("benign",0.0) else "BENIGN"
        # explanation via LLM if available
        summary = {"mode":"text","text": probs["phishing"], "probs": probs, "label": label}
        expl = make_llm_explanation(summary, None)
        return {"mode":"text","probs":probs,"label":label,"explanation": expl}
    except Exception as e:
        return {"error": f"Model loading or prediction failed: {str(e)}"}

@app.post("/api/url_check")
def url_check(req: URLCheckReq):
    try:
        tokenizer, model = load_url_model(URL_MODEL_DIR)
        probs = predict_url_probs(tokenizer, model, req.url)
        
        # Handle case where prediction fails
        if not probs:
            return {"error": "Failed to get prediction. Model may not be loaded properly."}
            
        # Use consistent verdict logic: highest probability wins (exact same as multimodal_infer.py)
        label = max(probs, key=lambda x: probs[x]) if probs else "unknown"
        
        summary = {"mode":"url","url": probs, "label": label}
        expl = make_llm_explanation({"url": url_multiclass_to_binary(probs), "probs": probs, "label": label}, None)
        return {"mode":"url","probs":probs,"label":label,"explanation": expl}
    except Exception as e:
        return {"error": f"Model loading or prediction failed: {str(e)}"}

@app.post("/api/image_check")
async def image_check(image_file: Optional[UploadFile] = File(None), image_path: Optional[str] = Form(None)):
    try:
        image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
        img_path_to_use = None
        if image_file:
            # Use Windows-compatible temp path
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image_file.filename}") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(await image_file.read())
            img_path_to_use = tmp_path
        elif image_path:
            img_path_to_use = image_path
        else:
            return {"error":"no image provided"}
            
        img_probs = predict_image_probs(image_model, img_path_to_use)
        
        # Handle case where prediction fails
        if not img_probs:
            return {"error": "Failed to get prediction. Model may not be loaded properly."}
            
        ocr_text = get_ocr_text_from_path(img_path_to_use)
        # Use consistent verdict logic: highest probability wins
        label = "PHISHING" if img_probs.get("phishing",0.0) > img_probs.get("benign",0.0) else "BENIGN"
        summary = {"mode":"image","img": img_probs, "label": label}
        expl = make_llm_explanation({"img": img_probs, "label": label}, ocr_text)
        return {"mode":"image","probs": img_probs, "ocr": ocr_text, "label": label, "explanation": expl}
    except Exception as e:
        return {"error": f"Model loading or prediction failed: {str(e)}"}

@app.post("/api/fusion")
async def fusion_endpoint(text: Optional[str] = Form(None), url: Optional[str] = Form(None), image_file: Optional[UploadFile] = File(None), image_path: Optional[str] = Form(None)):
    # load models as needed
    tokenizer_text = tokenizer_url = None
    text_model = url_model = image_model = None
    text_probs = url_probs = img_probs = None
    ocr_text = None

    if text:
        tokenizer_text, text_model = load_text_model(TEXT_MODEL_DIR)
        text_probs = predict_text_probs(tokenizer_text, text_model, text)
    if url:
        tokenizer_url, url_model = load_url_model(URL_MODEL_DIR)
        url_probs = predict_url_probs(tokenizer_url, url_model, url)
    img_path_to_use = None
    if image_file:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image_file.filename}") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(await image_file.read())
        img_path_to_use = tmp_path
    elif image_path:
        img_path_to_use = image_path
    if img_path_to_use:
        image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
        img_probs = predict_image_probs(image_model, img_path_to_use)
        ocr_text = get_ocr_text_from_path(img_path_to_use)
        # if OCR available and text_model loaded, compute ocr phishing prob
    ocr_prob = None
    if ocr_text and 'tokenizer_text' in locals() and tokenizer_text:
        tmp = predict_text_probs(tokenizer_text, text_model, ocr_text)
        ocr_prob = tmp["phishing"] if tmp else None

    text_p = text_probs["phishing"] if text_probs else None
    url_p = url_multiclass_to_binary(url_probs) if url_probs else None
    img_p = img_probs["phishing"] if img_probs else None

    fusion = fuse_all(text_p, url_p, img_p, ocr_prob, alpha=float(os.getenv("MM_FUSION_ALPHA", "0.80")))
    fusion.update({"raw_text": text_probs, "raw_url": url_probs, "raw_img": img_probs})
    expl = make_llm_explanation(fusion, ocr_text)
    return {"fusion": fusion, "ocr": ocr_text, "explanation": expl}

@app.post("/api/chat")
def chat(req: dict):
    # Accepts {"query": str, "image_path": optional}
    query = req.get("query", "")
    image_path = req.get("image_path", None)
    # Choose intent method (LLM or heuristic)
    if USE_LLM_INTENT:
        intent = classify_intent_llm(query, image_path)
    else:
        intent = heuristic_intent(query, image_path)
    # route
    if intent in ("chat", "unknown"):
        reply = friendly_chat_reply_llm(query)
        return {"intent": intent, "reply": reply}
    if intent == "url_check":
        url = extract_single_url(query)
        if not url:
            return {"error":"no url found in query"}
        tokenizer, model = load_url_model(URL_MODEL_DIR)
        probs = predict_url_probs(tokenizer, model, url)
        label = max(probs, key=probs.get) if probs else "unknown"
        expl = make_llm_explanation({"url": url_multiclass_to_binary(probs) if probs else None, "probs": probs, "label": label}, None)
        return {"intent":"url_check","url": url,"probs":probs,"label":label,"explanation":expl}
    if intent == "text_message":
        tokenizer, model = load_text_model(TEXT_MODEL_DIR)
        probs = predict_text_probs(tokenizer, model, query)
        label = "PHISHING" if probs and probs.get("phishing",0.0) > probs.get("benign",0.0) else "BENIGN"
        expl = make_llm_explanation({"text": probs["phishing"] if probs else None, "probs": probs, "label": label}, None)
        return {"intent": "text_message", "probs": probs, "label": label, "explanation": expl}
    if intent == "image_check":
        # require image_path or instruct upload
        if image_path:
            image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
            img_probs = predict_image_probs(image_model, image_path)
            ocr_text = get_ocr_text_from_path(image_path)
            label = "PHISHING" if img_probs and img_probs.get("phishing",0.0) > img_probs.get("benign",0.0) else "BENIGN"
            expl = make_llm_explanation({"img": img_probs, "label": label}, ocr_text)
            return {"intent":"image_check","probs": img_probs, "ocr": ocr_text, "label": label, "explanation": expl}
        return {"error":"image_check detected but no image_path provided. Use /api/image_check to upload."}

    return {"intent":"unknown"}

# MULTIPART CHAT ENDPOINT (text + image upload)
@app.post("/api/chat_multipart")
async def chat_multipart(query: str = Form(...), file: UploadFile = File(None)):
    # Select intent method - pass file object if image uploaded
    image_path = file.filename if file and file.filename else None
    if USE_LLM_INTENT:
        intent = classify_intent_llm(query, image_path)
    else:
        intent = heuristic_intent(query, image_path)

    # If chat or unknown → friendly response
    if intent in ("chat", "unknown"):
        return {
            "intent": intent,
            "reply": friendly_chat_reply_llm(query)
        }

    # If URL CHECK
    if intent == "url_check":
        url = extract_single_url(query)
        if not url:
            return {"error": "No URL found inside message"}

        tokenizer, model = load_url_model(URL_MODEL_DIR)
        probs = predict_url_probs(tokenizer, model, url)
        label = max(probs, key=probs.get)
        explanation = make_llm_explanation(
            {"url": url_multiclass_to_binary(probs), "probs": probs, "label": label},
            None
        )
        return {
            "intent": "url_check",
            "url": url,
            "probs": probs,
            "label": label,
            "explanation": explanation
        }

    # If TEXT MESSAGE
    if intent == "text_message":
        tokenizer, model = load_text_model(TEXT_MODEL_DIR)
        probs = predict_text_probs(tokenizer, model, query)
        label = "PHISHING" if probs["phishing"] > probs["benign"] else "BENIGN"
        explanation = make_llm_explanation(
            {"text": probs["phishing"], "probs": probs, "label": label},
            None
        )
        return {
            "intent": "text_message",
            "probs": probs,
            "label": label,
            "explanation": explanation
        }

    # If IMAGE CHECK
    if intent == "image_check":
        if not file:
            return {"error": "No image uploaded"}
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(await file.read())

        image_model = build_resnet50_for_loading(IMAGE_MODEL_PATH)
        probs = predict_image_probs(image_model, tmp_path)
        ocr_text = get_ocr_text_from_path(tmp_path)
        label = "PHISHING" if probs.get("phishing",0.0) > probs.get("benign",0.0) else "BENIGN"  # Use consistent logic

        explanation = make_llm_explanation(
            {"img": probs, "label": label},
            ocr_text
        )

        return {
            "intent": "image_check",
            "probs": probs,
            "ocr": ocr_text,
            "label": label,
            "explanation": explanation
        }

    return {"intent": "unknown"}
