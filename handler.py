import runpod
from diffusers import DiffusionPipeline
import torch
import os
import base64
from io import BytesIO

# Load Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Load the Hugging Face model with authentication
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
).to("cuda")

def handler(event):
    """
    Example request:
    {
      "input": {
        "prompt": "A cloud with a smiling face"
      }
    }
    """
    prompt = event.get("input", {}).get("prompt", None)
    if not prompt:
        return {"error": "No prompt provided"}

    # Generate image
    image = pipe(prompt).images[0]

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image_base64": image_b64}

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
