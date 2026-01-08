# Face Embedding API

FastAPI service for extracting 512-dimensional face embeddings using ArcFace model.

## Project Structure

```
.
├── main.py                     # FastAPI application
├── apitest.py                  # Test script
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker config
├── backbones/                  # Model backbones
└── work_dirs/ms1mv3_r100/
    └── model.pt               # Pre-trained model
```

## API Usage

### Request

```bash
curl -X POST http://localhost:8000/embed \
  -H 'x-api-key: tokentest' \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "0.1",
    "photos": ["BASE64_IMAGE_1", "BASE64_IMAGE_2"]
  }'
```

### Response

```json
{
  "version": "0.1",
  "descriptors": [
    [130.024, 124.330, ...],  // 512 dimensions
    [128.152, 126.894, ...]
  ]
}
```

### Test with Python

```bash
python apitest.py path/to/image.jpg
```

## Configuration

Edit `main.py` (lines 15-18):

```python
IMAGE_SIZE = (112, 112)
DEVICE = "cuda"  # or "cpu"
MODEL_PATH = "./models/model.pt"
API_KEY = "tokentest"  # Change for production
```

## Endpoints

- `POST /embed` - Extract embeddings (requires x-api-key header)
- `GET /health` - Health check
- `GET /` - API status

## Optimization

### Reduce Image Size (8GB → 2-3GB)

Use CPU-only PyTorch in `requirements.txt`:

```txt
# Replace torch lines with:
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1
torchvision==0.20.1
```

