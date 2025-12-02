# backend/app/models_loader.py
import os
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CACHE = {}

def load_text_model(text_dir: str):
    """Load BERT text classifier from local directory (absolute path enforced)."""
    key = f"text::{text_dir}"
    if key in _CACHE:
        return _CACHE[key]

    # Handle Windows paths and ensure proper path format
    abs_dir = os.path.abspath(text_dir).replace('\\', '/')
    print(f"Loading text model from: {abs_dir}")
    print(f"Directory exists: {os.path.exists(abs_dir)}")

    if not os.path.isdir(abs_dir):
        raise ValueError(f"[ERROR] Text model directory does NOT exist: {abs_dir}")

    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            abs_dir,
            local_files_only=True
        )

        model = BertForSequenceClassification.from_pretrained(
            abs_dir,
            local_files_only=True
        ).to(DEVICE)

        model.eval()
        _CACHE[key] = (tokenizer, model)
        print(f"✅ Text model loaded successfully from: {abs_dir}")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Failed to load text model: {str(e)}")
        raise ValueError(f"Failed to load text model from {abs_dir}: {str(e)}")



def load_url_model(url_dir: str):
    """Load DistilBERT URL classifier from local directory (absolute path enforced)."""
    key = f"url::{url_dir}"
    if key in _CACHE:
        return _CACHE[key]

    # Handle Windows paths and ensure proper path format
    abs_dir = os.path.abspath(url_dir).replace('\\', '/')
    print(f"Loading URL model from: {abs_dir}")
    print(f"Directory exists: {os.path.exists(abs_dir)}")

    if not os.path.isdir(abs_dir):
        raise ValueError(f"[ERROR] URL model directory does NOT exist: {abs_dir}")

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            abs_dir,
            local_files_only=True
        )

        model = DistilBertForSequenceClassification.from_pretrained(
            abs_dir,
            local_files_only=True
        ).to(DEVICE)

        model.eval()
        _CACHE[key] = (tokenizer, model)
        print(f"✅ URL model loaded successfully from: {abs_dir}")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Failed to load URL model: {str(e)}")
        raise ValueError(f"Failed to load URL model from {abs_dir}: {str(e)}")

def build_resnet50_for_loading(state_dict_path: str):
    key = f"image::{state_dict_path}"
    if key in _CACHE:
        return _CACHE[key]
    
    # Handle Windows paths and ensure proper path format
    abs_path = os.path.abspath(state_dict_path).replace('\\', '/')
    print(f"Loading image model from: {abs_path}")
    print(f"File exists: {os.path.exists(abs_path)}")
    
    if not os.path.isfile(abs_path):
        raise ValueError(f"[ERROR] Image model file does NOT exist: {abs_path}")
    
    try:
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, 2))
        
        print(f"Loading state dict from: {abs_path}")
        sd = torch.load(abs_path, map_location=DEVICE)
        
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
        _CACHE[key] = model
        print(f"✅ Image model loaded successfully from: {abs_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load image model: {str(e)}")
        raise ValueError(f"Failed to load image model from {abs_path}: {str(e)}")

def get_cache_keys():
    return list(_CACHE.keys())
