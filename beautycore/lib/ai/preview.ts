import 'server-only';

import type { StyleCategory } from './types';

/**
 * Image-generation providers.
 *
 * Preview generation is deliberately behind an interface: as of this build no
 * provider is available on the project's free-tier keys (HuggingFace retired
 * SDXL serverless; Gemini image models return quota `limit: 0` without
 * billing; Replicate needs a paid token). Analysis still works, so the app is
 * useful today and a provider can be enabled later without touching routes.
 *
 * To enable one, set PREVIEW_PROVIDER in .env.local and add the matching key.
 */

export interface GenerateRequest {
  prompt: string;
  styleType: StyleCategory;
  /** Base64 source photo for image-to-image, when the provider supports it. */
  sourceImage?: { base64: string; mimeType: string };
}

export interface GenerateSuccess {
  ok: true;
  /** Data URL or hosted URL of the generated preview. */
  imageUrl: string;
  provider: string;
}

export interface GenerateUnavailable {
  ok: false;
  reason: 'not_configured' | 'quota' | 'provider_error';
  /** Shown to the user — must be plain and non-technical. */
  message: string;
  provider: string;
}

export type GenerateResult = GenerateSuccess | GenerateUnavailable;

export interface PreviewProvider {
  name: string;
  /** False when required credentials are absent. */
  isConfigured(): boolean;
  generate(req: GenerateRequest): Promise<GenerateResult>;
}

// ─── Stub (default) ─────────────────────────────────────────────────────────

/**
 * Always reports unavailable, with an honest explanation. Keeps the full
 * upload → analyse → (attempt preview) flow exercisable end to end.
 */
const stubProvider: PreviewProvider = {
  name: 'stub',
  isConfigured: () => true,
  async generate(): Promise<GenerateResult> {
    return {
      ok: false,
      reason: 'not_configured',
      message:
        'Preview generation is not enabled yet. Your style analysis and recommendations above are live — image previews need a generation provider to be configured.',
      provider: 'stub',
    };
  },
};

// ─── Gemini (requires billing) ──────────────────────────────────────────────

const geminiProvider: PreviewProvider = {
  name: 'gemini',
  isConfigured: () => Boolean(process.env.GEMINI_API_KEY),

  async generate({ prompt, sourceImage }): Promise<GenerateResult> {
    if (!this.isConfigured()) {
      return {
        ok: false,
        reason: 'not_configured',
        message: 'Image generation is not configured.',
        provider: 'gemini',
      };
    }

    try {
      const { GoogleGenAI } = await import('@google/genai');
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      const model = process.env.GEMINI_IMAGE_MODEL ?? 'gemini-2.5-flash-image';

      // With a source photo this becomes an edit rather than a fresh render,
      // which is what keeps the client's own hand/face in the preview.
      const parts: Array<Record<string, unknown>> = [];
      if (sourceImage) {
        parts.push({
          inlineData: { mimeType: sourceImage.mimeType, data: sourceImage.base64 },
        });
        parts.push({
          text: `Edit this photo to show the following style, keeping the person's actual hand/face structure, skin tone, lighting, and background unchanged. Only change the styled element. Style: ${prompt}`,
        });
      } else {
        parts.push({ text: prompt });
      }

      const res = await ai.models.generateContent({
        model,
        contents: [{ role: 'user', parts }],
        config: { responseModalities: ['IMAGE'] },
      });

      const out = res.candidates?.[0]?.content?.parts?.find((p) => p.inlineData);
      if (!out?.inlineData?.data) {
        return {
          ok: false,
          reason: 'provider_error',
          message: 'The image provider did not return an image. Please try again.',
          provider: 'gemini',
        };
      }

      const mime = out.inlineData.mimeType ?? 'image/png';
      return {
        ok: true,
        imageUrl: `data:${mime};base64,${out.inlineData.data}`,
        provider: 'gemini',
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isQuota = msg.includes('429') || msg.toLowerCase().includes('quota');

      console.error('[preview/gemini]', msg.slice(0, 400));

      return {
        ok: false,
        reason: isQuota ? 'quota' : 'provider_error',
        message: isQuota
          ? 'Image generation is over its quota. This model requires a billed Google Cloud account.'
          : 'Preview generation failed. Please try again in a moment.',
        provider: 'gemini',
      };
    }
  },
};

// ─── LoRA (Colab-hosted, via tunnel) ────────────────────────────────────────

/**
 * Talks to a self-hosted LoRA inference server — typically the Colab notebook
 * in scripts/colab_lora_server.py exposed through a cloudflared tunnel.
 *
 * Contract:
 *   POST {LORA_ENDPOINT_URL}/generate
 *   { prompt, image?, prompt_strength }  ->  { image: "<base64 png>" }
 *
 * The tunnel URL changes on every notebook restart, so failures here are
 * expected and get a specific message rather than a generic error.
 */
const loraProvider: PreviewProvider = {
  name: 'lora',
  isConfigured: () => Boolean(process.env.LORA_ENDPOINT_URL),

  async generate({ prompt, styleType, sourceImage }): Promise<GenerateResult> {
    const base = process.env.LORA_ENDPOINT_URL?.replace(/\/+$/, '');
    if (!base) {
      return {
        ok: false,
        reason: 'not_configured',
        message:
          'The style model is not connected. Start the Colab notebook and set LORA_ENDPOINT_URL in .env.local.',
        provider: 'lora',
      };
    }

    // Nails need the hand kept intact, so change less of the source image.
    // Hair can move further from the original without looking wrong.
    const promptStrength = styleType === 'nails' ? 0.5 : 0.65;

    // Cold start loads SDXL onto the GPU, which is slow the first time.
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 180_000);

    try {
      const res = await fetch(`${base}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          prompt,
          image: sourceImage
            ? `data:${sourceImage.mimeType};base64,${sourceImage.base64}`
            : undefined,
          prompt_strength: promptStrength,
        }),
      });

      if (!res.ok) {
        console.error('[preview/lora] HTTP', res.status);
        return {
          ok: false,
          reason: 'provider_error',
          message: `The style model returned an error (${res.status}). Check the Colab notebook is still running.`,
          provider: 'lora',
        };
      }

      const data = (await res.json()) as { image?: string; error?: string };

      if (data.error) {
        console.error('[preview/lora]', data.error);
        return {
          ok: false,
          reason: 'provider_error',
          message: 'The style model failed to render this look. Please try another style.',
          provider: 'lora',
        };
      }

      if (!data.image) {
        return {
          ok: false,
          reason: 'provider_error',
          message: 'The style model returned no image. Please try again.',
          provider: 'lora',
        };
      }

      const img = data.image.startsWith('data:')
        ? data.image
        : `data:image/png;base64,${data.image}`;

      return { ok: true, imageUrl: img, provider: 'lora' };
    } catch (err) {
      const aborted = err instanceof Error && err.name === 'AbortError';
      console.error('[preview/lora]', err instanceof Error ? err.message : err);

      return {
        ok: false,
        reason: 'provider_error',
        message: aborted
          ? 'The style model took too long to respond. The Colab GPU may be loading — try again in a minute.'
          : 'Could not reach the style model. Check the Colab notebook is running and LORA_ENDPOINT_URL is current.',
        provider: 'lora',
      };
    } finally {
      clearTimeout(timeout);
    }
  },
};

// ─── Selection ──────────────────────────────────────────────────────────────

const providers: Record<string, PreviewProvider> = {
  stub: stubProvider,
  gemini: geminiProvider,
  lora: loraProvider,
};

export function getPreviewProvider(): PreviewProvider {
  const name = (process.env.PREVIEW_PROVIDER ?? 'stub').toLowerCase();
  const provider = providers[name];

  if (!provider) {
    console.warn(
      `[preview] Unknown PREVIEW_PROVIDER "${name}". Falling back to stub. Valid: ${Object.keys(providers).join(', ')}`
    );
    return stubProvider;
  }
  return provider;
}
