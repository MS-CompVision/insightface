import io
import base64
import torch
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from backbones import get_model

# -------------------------
# Configuration
# -------------------------
IMAGE_SIZE = (112, 112)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./models/model.pt"
API_KEY = "tokentest"

# Global model variable
model = None


# -------------------------
# Pydantic Models
# -------------------------
class EmbeddingRequest(BaseModel):
    version: str
    photos: List[str]


class EmbeddingResponse(BaseModel):
    version: str
    descriptors: List[List[float]]


# -------------------------
# Utilities
# -------------------------
def load_backbone(model_path: str):
    """Load the face recognition backbone model"""
    backbone = get_model(
        "r100",
        dropout=0.0,
        fp16=True,
        num_features=512
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone


def base64_to_tensor(base64_str: str):
    """Convert base64 image string to preprocessed tensor"""
    try:
        # Decode base64 string
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot decode image")

        # Preprocess image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = (img_tensor - 127.5) / 128.0  # ArcFace normalization
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        return img_tensor
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def get_embedding(model, img_tensor):
    """Extract normalized embedding from image tensor"""
    with torch.no_grad():
        emb = model(img_tensor)
        emb = torch.nn.functional.normalize(emb)
    return emb.squeeze(0)


# -------------------------
# FastAPI Lifespan
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Using device: {DEVICE}")
    model = load_backbone(MODEL_PATH)
    print("Model loaded successfully!")
    yield
    # Shutdown: Cleanup if needed
    model = None


# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Face Embedding API",
    description="API for extracting face embeddings using ArcFace",
    version="0.1",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": DEVICE,
        "version": "0.1"
    }


@app.post("/arcface", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    x_api_key: Optional[str] = Header(None)
):
    """
    Extract face embeddings from base64-encoded images
    
    Args:
        request: JSON body containing version and list of base64-encoded images
        x_api_key: API key for authentication
    
    Returns:
        JSON response with version and list of embedding descriptors
    """
    # API Key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not request.photos:
        raise HTTPException(status_code=400, detail="No photos provided")
    
    descriptors = []
    
    try:
        for idx, base64_img in enumerate(request.photos):
            # Preprocess image
            img_tensor = base64_to_tensor(base64_img)
            
            # Get embedding
            embedding = get_embedding(model, img_tensor)
            
            # Convert to list of floats
            descriptor = embedding.cpu().numpy().tolist()
            descriptors.append(descriptor)
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    return EmbeddingResponse(
        version=request.version,
        descriptors=descriptors
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
