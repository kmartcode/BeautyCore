# ============================================================================
#  BeautyCore — LoRA inference server for Google Colab
# ============================================================================
#  Serves a trained LoRA over HTTP so the Next.js app can call it.
#
#  HOW TO USE
#  ----------
#  1. Runtime > Change runtime type > T4 GPU (or better). Then Run all.
#  2. Upload your trained .safetensors, or set LORA_PATH to a Drive path.
#  3. Copy the printed https://....trycloudflare.com URL.
#  4. In beautycore/.env.local set:
#         PREVIEW_PROVIDER=lora
#         LORA_ENDPOINT_URL=https://....trycloudflare.com
#     Restart `npm run dev`.
#
#  IMPORTANT
#  ---------
#  * The URL changes every time this notebook restarts. Re-paste it each session.
#  * Keep this tab open and the runtime alive during your demo.
#  * Free-tier Colab reclaims GPUs when idle — click something occasionally.
# ============================================================================

# --- Cell 1: install -------------------------------------------------------
!pip install -q diffusers transformers accelerate peft safetensors fastapi uvicorn nest-asyncio pyngrok pillow
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared


# --- Cell 2: load the pipeline --------------------------------------------
import torch, base64, io, threading, subprocess, re, time
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH  = "/content/nails_lora.safetensors"   # <-- your trained file
LORA_SCALE = 0.8                                  # 0.6-1.0; higher = stronger style

print("Loading base model (a few minutes on first run)...")
img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# Text-to-image shares the same weights — no extra VRAM.
txt2img = StableDiffusionXLPipeline(**img2img.components).to("cuda")

import os
if os.path.exists(LORA_PATH):
    img2img.load_lora_weights(LORA_PATH)
    print(f"✓ LoRA loaded from {LORA_PATH}")
else:
    print(f"⚠ No LoRA at {LORA_PATH} — serving BASE model only.")
    print("  Upload your .safetensors there, then re-run this cell.")

img2img.set_progress_bar_config(disable=True)
txt2img.set_progress_bar_config(disable=True)
print("✓ Pipeline ready")


# --- Cell 3: the API -------------------------------------------------------
import nest_asyncio, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

nest_asyncio.apply()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

NEGATIVE = ("ugly, disfigured, deformed, blurry, low quality, watermark, text, "
            "extra fingers, missing fingers, mutated hands, bad anatomy")


class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None          # base64, with or without data: prefix
    prompt_strength: float = 0.5         # img2img only: 0=keep original, 1=ignore it
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


def _decode(b64: str) -> Image.Image:
    if "," in b64:                        # strip "data:image/png;base64,"
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    # SDXL wants dimensions on a multiple of 8, capped for VRAM.
    w, h = img.size
    scale = min(1024 / max(w, h), 1.0)
    w, h = (int(w * scale) // 8 * 8, int(h * scale) // 8 * 8)
    return img.resize((max(w, 512), max(h, 512)), Image.LANCZOS)


@app.get("/health")
def health():
    return {"status": "ok", "lora": os.path.exists(LORA_PATH), "base": BASE_MODEL}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Contract the BeautyCore provider expects: {image: "<base64 png>"}."""
    try:
        opts = dict(
            prompt=req.prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            cross_attention_kwargs={"scale": LORA_SCALE},
        )

        if req.image:
            # img2img — preserves the client's real hand/face structure.
            out = img2img(
                image=_decode(req.image),
                strength=max(0.1, min(req.prompt_strength, 0.95)),
                **opts,
            ).images[0]
        else:
            out = txt2img(width=1024, height=1024, **opts).images[0]

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode()}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# --- Cell 4: expose it -----------------------------------------------------
threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
    daemon=True,
).start()
time.sleep(3)

proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)

print("Opening tunnel...")
for line in proc.stdout:
    m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
    if m:
        print("\n" + "=" * 68)
        print("  PUBLIC URL:", m.group(0))
        print("=" * 68)
        print("\n  Put this in beautycore/.env.local:\n")
        print("    PREVIEW_PROVIDER=lora")
        print(f"    LORA_ENDPOINT_URL={m.group(0)}")
        print("\n  Then restart `npm run dev`. Keep this notebook running.")
        print("=" * 68)
        break
