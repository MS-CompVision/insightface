import base64
import requests
import json

# Read and encode image
with open("face.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/arcface",
    headers={
        "x-api-key": "tokentest",
        "Content-Type": "application/json"
    },
    json={
        "version": "0.1",
        "photos": [img_base64]
    }
)

response_data = response.json()
#print(response_data)

# Print number of arrays and size of each
descriptors = response_data.get('descriptors', [])
print(f"\nNumber of embeddings: {len(descriptors)}")
for i, desc in enumerate(descriptors):
    print(f"Embedding {i+1}: {len(desc)} dimensions")
