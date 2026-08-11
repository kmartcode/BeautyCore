# BeautyCore — Andrea's Salon Management System

An AI-assisted salon platform: clients upload a photo, Google Gemini analyses
their current hair or nails and recommends styles grounded in what it actually
sees. Staff manage bookings, inventory, and finances from role-scoped portals.

Built with Next.js 16 (App Router), React 19, TypeScript (strict), Tailwind 4,
Drizzle ORM, and Neon PostgreSQL.

---

## Quick start

```bash
npm install
cp .env.local.example .env.local   # then fill in your keys
npm run db:push                    # create tables
npm run db:seed                    # load demo data
npm run dev                        # http://localhost:3000
```

`db:seed` is idempotent — it clears and reloads demo rows, so you can run it
whenever you want a clean slate.

## Demo accounts

| Role | Email | Password |
|---|---|---|
| Admin | `admin@andreas.com` | `admin123` |
| Stylist | `lara@andreas.com` | `stylist123` |
| Client | `maria@email.com` | `client123` |

Seeded with 9 users, 14 appointments, 18 inventory products, and 5 AI records.

---

## Architecture

```
app/
  (public)/          home, services, about, contact, booking
  (auth)/            login, register
  client/            dashboard, ai-advisor, nail/hair studio, appointments, profile
  admin/             dashboard, appointments, clients, inventory, finance,
                     market-trends, security
  stylist/           dashboard, appointments, clients, services, profile
  api/               auth, appointments, inventory, users, profile,
                     analyze, generate, generations, admin/stats
components/          ui.tsx (shared primitives), sidebars, navbar, footer
context/             AuthContext
db/                  schema.ts, index.ts, seed.ts
lib/                 auth.ts, services.ts, ai/
scripts/             colab_lora_server.py
```

**Auth** — bcrypt password hashing, JWT in an HttpOnly cookie (`jose`), 7-day
expiry. `getSession()` / `requireRole()` guard every protected route handler.
Role checks live server-side; `ProtectedRoute` only handles redirects.

**Data scoping** — the same `/api/appointments` endpoint returns different rows
per role: clients see their own, stylists see those assigned to them, admins see
everything. Scoping happens in the query, not the UI.

---

## The AI pipeline

Two stages, deliberately separated:

1. **Analysis** (`/api/analyze`) — Gemini vision reads the photo and returns the
   current shape, colour, and condition, plus three recommendations with colour
   palettes, design details, reasoning, and a generation prompt. This is fully
   live and the part your capstone demo rests on.

2. **Preview** (`/api/generate`) — renders a recommendation onto the client's
   photo. Provider-swappable via `PREVIEW_PROVIDER`:

   | Value | Behaviour |
   |---|---|
   | `stub` *(default)* | No image. Analysis still works; the UI explains why. |
   | `lora` | Calls your own LoRA over HTTP. See below. |
   | `gemini` | Google image models — **needs a billed Google Cloud account.** |

Every generation is persisted to `ai_generations` regardless of whether a
preview rendered, so the analysis is never lost.

### Wiring up the LoRA

The trained model is a **Stable Diffusion 1.5** LoRA (rank 32, 512px, 1500
steps), trained with the diffusers `train_text_to_image_lora.py` script. It is
UNet-only — no text encoder weights.

`scripts/colab_lora_server.py` turns a Colab notebook into an image-to-image
endpoint for it:

1. Open the script in Colab, set the runtime to **T4 GPU**
2. Upload `pytorch_lora_weights.safetensors` (or mount Drive and point
   `LORA_PATH` at the model folder), then run the cells in order
3. **Run the smoke-test cell and look at the output.** Training used
   `validation_prompt: null`, so no sample image was produced during those 1500
   steps — the smoke test is the first look anyone gets. It sweeps LoRA scale
   and asserts the output actually differs with the LoRA off vs on, which
   catches a LoRA that loads but never reaches the UNet.
4. Copy the printed `https://….trycloudflare.com` URL
5. Add to `.env.local` and restart:
   ```
   PREVIEW_PROVIDER=lora
   LORA_ENDPOINT_URL=https://your-tunnel.trycloudflare.com
   ```

The tunnel URL changes on every Colab restart — re-paste it each session.

Notes worth knowing:

- **It's SD 1.5, not SDXL.** Serving it on an SDXL base fails outright: this
  LoRA's cross-attention layers are 768-dim against SDXL's 2048.
- **The base model id in `hparams.yml` is dead.** Runway deleted their
  HuggingFace repos, so `runwayml/stable-diffusion-v1-5` now 404s. The script
  tries the surviving mirrors of the same weights in order.
- **The NSFW safety checker is disabled deliberately.** SD 1.5's checker
  false-positives on close-up skin — hands and faces, i.e. this app's only
  input — and returns a solid black image instead of an error.
- **`LORA_STYLE_SCOPE` defaults to `nails`,** matching the model that exists.
  A nails LoRA applied to a hair photo produces nonsense, so hair requests get
  an honest "not available yet" instead. Widen it to `hair,nails` once a hair
  LoRA is trained.
- **Checkpoints at 500/1000/1500 steps** are in the model folder. 1500 is the
  default; warmup ran for the first 500, so `checkpoint-500` is undertrained.
  If 1500 looks overcooked, compare against `checkpoint-1000`.

---

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server (Turbopack) |
| `npm run build` | Production build |
| `npm run start` | Serve the build |
| `npm run lint` | ESLint |
| `npm run db:push` | Push schema to Neon |
| `npm run db:seed` | Reset and load demo data |
| `npm run db:studio` | Drizzle Studio (browse the DB) |

## Notes

- `_legacy_src/` holds the original Vite + React Router code, kept because this
  project isn't under version control. Nothing imports from it; delete it once
  you're confident in the migration.
- Prices are stored as whole pesos in `integer` columns — no floating-point
  rounding on money.
- Services marked *Mula* (from) vary by hair length and condition.
