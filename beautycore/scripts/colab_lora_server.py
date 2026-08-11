# ============================================================================
#  BeautyCore — LoRA inference server for Google Colab  (Stable Diffusion 1.5)
# ============================================================================
#  Serves the trained BeautyCore LoRA over HTTP so the Next.js app can call it.
#
#  This is built for the LoRA in BeautyCore_Model/, whose training config was:
#
#      pretrained_model_name_or_path : runwayml/stable-diffusion-v1-5
#      resolution                    : 512
#      rank                          : 32
#      max_train_steps               : 1500   (checkpoints at 500/1000/1500)
#      validation_prompt             : null   <-- never validated during training
#
#  SD 1.5, NOT SDXL. Serving it on an SDXL base fails outright: this LoRA's
#  cross-attention layers are 768-dim, SDXL's are 2048.
#
#  HOW TO USE
#  ----------
#  1. Runtime > Change runtime type > T4 GPU. Then run the cells in order.
#  2. Upload pytorch_lora_weights.safetensors, or mount Drive and point
#     LORA_PATH at BeautyCore_Model/.
#  3. Run Cell 3 (smoke test) and LOOK AT THE OUTPUT before wiring up the app.
#     Nothing was validated during training, so this is the first time anyone
#     sees what this model actually produces.
#  4. Copy the printed https://....trycloudflare.com URL into .env.local:
#         PREVIEW_PROVIDER=lora
#         LORA_ENDPOINT_URL=https://....trycloudflare.com
#     Restart `npm run dev`.
#
#  IMPORTANT
#  ---------
#  * The URL changes every time this notebook restarts. Re-paste it each session.
#  * Keep this tab open and the runtime alive during your demo.
#  * Free-tier Colab reclaims idle GPUs — interact with the page occasionally.
# ============================================================================


# --- Cell 1: install -------------------------------------------------------
!pip install -q "diffusers>=0.31" transformers accelerate peft safetensors \
                fastapi "uvicorn[standard]" nest-asyncio pillow
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared


# --- Cell 2: load the pipeline ---------------------------------------------
import os, io, re, time, base64, threading, subprocess
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

# Point this at the .safetensors file, or at a directory containing one.
# To compare checkpoints, swap in e.g. ".../BeautyCore_Model/checkpoint-1000".
LORA_PATH = "/content/pytorch_lora_weights.safetensors"

# How strongly the LoRA is applied. 0 = base model, 1 = full strength.
DEFAULT_LORA_SCALE = 0.8

# img2img input size. SD 1.5 trains at 512; pushing far past that invites
# duplicated fingers and doubled subjects. 640 keeps a little more nail
# detail while staying close to native. Lower it to 512 if you see artefacts.
MAX_SIDE = 640

# runwayml deleted their Stable Diffusion repos from HuggingFace in 2024, so the
# id recorded in hparams.yml now 404s. These are the surviving mirrors of the
# same base weights; the first one that loads wins.
BASE_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
]

img2img = None
BASE_MODEL = None
for candidate in BASE_CANDIDATES:
    # These mirrors publish F32 weights with no fp16 variant, so do NOT pass
    # variant="fp16" — it raises. torch_dtype casts at load time instead.
    # Try safetensors first, then fall back for any mirror that only has .bin.
    for use_st in (True, False):
        try:
            print(f"Trying base model: {candidate} (safetensors={use_st}) ...")
            img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                candidate,
                torch_dtype=torch.float16,
                use_safetensors=use_st,
                # SD 1.5 ships an NSFW checker that false-positives on close-up
                # skin — hands and faces, i.e. exactly this app's input — and
                # returns a solid black image. Off, or previews mysteriously blank.
                safety_checker=None,
                requires_safety_checker=False,
            ).to("cuda")
            BASE_MODEL = candidate
            print(f"✓ Loaded {candidate}")
            break
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {str(e)[:160]}")
    if img2img is not None:
        break

if img2img is None:
    raise RuntimeError(
        "Could not load any SD 1.5 base model. Check your internet connection, "
        "or pass a local path / another mirror in BASE_CANDIDATES."
    )

# Text-to-image reuses the same weights, so this costs no extra VRAM. It is
# what the smoke test uses to show what the LoRA learned in isolation.
txt2img = StableDiffusionPipeline(**img2img.components)
txt2img.set_progress_bar_config(disable=True)
img2img.set_progress_bar_config(disable=True)


def _resolve_lora(path: str) -> str | None:
    """Accept either a .safetensors file or a directory holding one."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cand = os.path.join(path, "pytorch_lora_weights.safetensors")
        if os.path.isfile(cand):
            return cand
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors"):
                return os.path.join(path, f)
    return None


LORA_FILE = _resolve_lora(LORA_PATH)
LORA_READY = False

if LORA_FILE:
    try:
        img2img.load_lora_weights(LORA_FILE)
        LORA_READY = True
        print(f"✓ LoRA loaded from {LORA_FILE}")
    except Exception as e:
        print(f"✗ LoRA failed to load: {type(e).__name__}: {e}")
        print("  This LoRA uses the legacy diffusers 'lora.down/lora.up' key")
        print("  naming. If load failed on key names, try: pip install -U diffusers")
else:
    print(f"⚠ No .safetensors found at {LORA_PATH} — serving BASE model only.")
    print("  Upload the file or fix the path, then re-run this cell.")


def set_lora_scale(scale: float):
    """
    Apply a LoRA weight. diffusers changed this API across versions, so try the
    modern PEFT path first and report whether the caller must fall back to
    passing cross_attention_kwargs at call time.
    """
    if not LORA_READY:
        return None
    try:
        names = img2img.get_active_adapters()
        if names:
            img2img.set_adapters(names, adapter_weights=[scale] * len(names))
            return "adapters"
    except Exception:
        pass
    return "cross_attention_kwargs"


SCALE_MODE = set_lora_scale(DEFAULT_LORA_SCALE)
print(f"LoRA scale mechanism: {SCALE_MODE or 'n/a (no LoRA)'}")


def _call_kwargs(scale: float) -> dict:
    """Per-request kwargs needed to honour `scale` on this diffusers version."""
    if not LORA_READY:
        return {}
    if SCALE_MODE == "adapters":
        set_lora_scale(scale)
        return {}
    return {"cross_attention_kwargs": {"scale": scale}}


NEGATIVE = (
    "ugly, disfigured, deformed, blurry, low quality, watermark, text, "
    "extra fingers, missing fingers, fused fingers, mutated hands, bad anatomy"
)


def _decode(b64: str) -> Image.Image:
    """Base64 (with or without data: prefix) -> RGB image sized for SD 1.5."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    w, h = img.size
    scale = min(MAX_SIDE / max(w, h), 1.0)
    w, h = max(int(w * scale) // 8 * 8, 384), max(int(h * scale) // 8 * 8, 384)
    return img.resize((w, h), Image.LANCZOS)


def render(prompt, image=None, strength=0.5, scale=DEFAULT_LORA_SCALE,
           steps=40, guidance=7.5, seed=None):
    """Single entry point used by both the smoke test and the API."""
    gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    opts = dict(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        guidance_scale=guidance,
        generator=gen,
        **_call_kwargs(scale),
    )

    if image is not None:
        # img2img runs steps*strength actual denoising steps, so keep `steps`
        # generous or low-strength renders come out under-cooked.
        return img2img(
            image=image,
            strength=max(0.1, min(strength, 0.95)),
            num_inference_steps=steps,
            **opts,
        ).images[0]

    return txt2img(width=512, height=512, num_inference_steps=steps, **opts).images[0]


print("✓ Pipeline ready")


# --- Cell 3: SMOKE TEST — run this and actually look at it -----------------
#  Training used validation_prompt: null, so no sample was ever generated
#  during those 1500 steps. Two things can go wrong silently:
#    1. the LoRA loads but is never applied (output == base model)
#    2. the LoRA trained on captions nobody wrote down, so plain prompts
#       don't activate the style
#  This cell catches both. Compare the columns: if they look identical, the
#  LoRA is not reaching the UNet and previews will be indistinguishable from
#  stock SD 1.5.

from IPython.display import display

TEST_PROMPT = "close-up photo of manicured fingernails, glossy almond nails, salon quality"
SEED = 42


def grid(images, labels, cell=320):
    cols = len(images)
    canvas = Image.new("RGB", (cell * cols, cell + 28), "white")
    for i, im in enumerate(images):
        canvas.paste(im.resize((cell, cell), Image.LANCZOS), (i * cell, 28))
    try:
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        for i, lab in enumerate(labels):
            d.text((i * cell + 8, 8), lab, fill="black")
    except Exception:
        print(" | ".join(labels))
    return canvas


if LORA_READY:
    print("Rendering LoRA scale sweep (same seed, so differences are the LoRA)...")
    scales = [0.0, 0.5, 0.8, 1.0]
    imgs = [render(TEST_PROMPT, scale=s, seed=SEED, steps=30) for s in scales]
    display(grid(imgs, [f"scale {s}" for s in scales]))

    # Numeric check — eyeballing can miss a subtle no-op.
    import numpy as np
    base_arr = np.asarray(imgs[0], dtype=np.float32)
    full_arr = np.asarray(imgs[-1], dtype=np.float32)
    delta = float(np.abs(base_arr - full_arr).mean())
    print(f"\nMean pixel difference, scale 0.0 vs 1.0: {delta:.2f}")
    if delta < 1.0:
        print("✗ FAIL — the LoRA is NOT affecting output. Previews would be")
        print("  identical to stock SD 1.5. Check the load step above.")
    else:
        print("✓ PASS — the LoRA measurably changes the output.")
else:
    print("Skipped: no LoRA loaded.")


# Optional: upload one real client photo as /content/test.jpg to preview the
# actual img2img path the app uses. This is the more important test of the two.
if LORA_READY and os.path.exists("/content/test.jpg"):
    src = _decode(base64.b64encode(open("/content/test.jpg", "rb").read()).decode())
    strengths = [0.35, 0.5, 0.65]
    outs = [render(TEST_PROMPT, image=src, strength=s, seed=SEED) for s in strengths]
    print("\nimg2img strength sweep — pick the lowest value that still restyles:")
    display(grid([src] + outs, ["original"] + [f"strength {s}" for s in strengths]))
else:
    print("\n(Upload /content/test.jpg to preview the img2img path the app uses.)")


# --- Cell 4: the API -------------------------------------------------------
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


class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None       # base64, with or without data: prefix
    prompt_strength: float = 0.5      # img2img: 0 = keep original, 1 = ignore it
    num_inference_steps: int = 40
    guidance_scale: float = 7.5
    lora_scale: float = DEFAULT_LORA_SCALE


@app.get("/health")
def health():
    return {
        "status": "ok",
        "base": BASE_MODEL,
        "lora": LORA_READY,
        "lora_file": LORA_FILE,
        "scale_mode": SCALE_MODE,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """Contract the BeautyCore provider expects: {image: "<base64 png>"}."""
    try:
        out = render(
            req.prompt,
            image=_decode(req.image) if req.image else None,
            strength=req.prompt_strength,
            scale=req.lora_scale,
            steps=req.num_inference_steps,
            guidance=req.guidance_scale,
        )
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode()}

    except Exception as e:
        # Returned as 200 with an error field; the Next.js provider surfaces
        # this as a friendly message rather than a stack trace.
        return {"error": f"{type(e).__name__}: {e}"}


# --- Cell 5: expose it -----------------------------------------------------
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
