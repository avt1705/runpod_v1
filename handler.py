import runpod
import torch
import base64
from io import BytesIO
from diffusers import FluxPipeline

# Load the model in bfloat16 (Required for FLUX stability and GQA support)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

def handler(event):
    # Get user input prompt
    prompt = event.get("input", {}).get("prompt", None)

    if not prompt:
        return {"error": "No prompt provided in request."}

    # Generate image
    # Note: FLUX models often require specific height/width and steps for best results
    image = pipe(
        prompt, 
        height=1024, 
        width=1024, 
        guidance_scale=3.5, 
        num_inference_steps=50
    ).images[0]

    # Save and encode
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": image_b64
    }

runpod.serverless.start({"handler": handler})
