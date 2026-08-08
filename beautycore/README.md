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

### Wiring up a LoRA

`scripts/colab_lora_server.py` turns a Colab notebook into an image-to-image
endpoint for a trained LoRA:

1. Train your LoRA, export `.safetensors`
2. Paste the script into Colab (**T4 GPU**), upload the weights, Run all
3. Copy the printed `https://….trycloudflare.com` URL
4. Add to `.env.local` and restart:
   ```
   PREVIEW_PROVIDER=lora
   LORA_ENDPOINT_URL=https://your-tunnel.trycloudflare.com
   ```

The tunnel URL changes on every Colab restart — re-paste it each session.

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
