# Sea Turtle ID API — Frontend Guide

Base URL: `https://seraffins-machine-learning-workflow.onrender.com`
Auth: every `/v1` request needs the header `X-API-Key: <key>`.
All bodies are JSON.

> **Call this API from your backend only.** The key is a server secret: never put it in browser
> JavaScript or a mobile app. Browser (CORS) calls are disabled on purpose.

## The flow

```
1. Frontend uploads the user's photos to its own R2 (as it does today)
2. POST /v1/identify  { uid, images: { whole, left, right, front, shell, rear } }
   -> waits for the models, then returns the result in the same response (200)
```

A warm run takes a few seconds. The first run after an idle period can take 1–2 minutes (GPU
cold start), so **set your HTTP client timeout to at least 5 minutes** and show a progress state.
If processing takes longer than 4 minutes you get `504` with the `job_id`; fetch the full result
later with `GET /v1/jobs/{job_id}`.

**Image links:** https only, from the agreed R2 host(s). Presigned links must stay valid for at
least 30 minutes. Allowed formats: JPEG, PNG, WebP (convert HEIC on the device).

---

## 1. Start identification

`POST /v1/identify` → `202 Accepted`

```json
{
  "uid": "user-42",
  "images": {
    "whole": "https://pub-xxxx.r2.dev/u42/whole.jpg",
    "left":  "https://pub-xxxx.r2.dev/u42/left.jpg",
    "right": "https://pub-xxxx.r2.dev/u42/right.jpg",
    "front": "https://pub-xxxx.r2.dev/u42/front.jpg",
    "shell": "https://pub-xxxx.r2.dev/u42/shell.jpg",
    "rear":  ""
  }
}
```

- `uid` (required): your user's id. Every job, result and webhook carries it back.
- `images`: one R2 link per slot. Send `""` (or leave the key out) for a photo the user didn't take.
  At least one is required. Unknown slot names are rejected.

| Slot | What to photograph | Used for |
|---|---|---|
| `whole` | Whole turtle | Species |
| `left` | **Left side of the head, close-up** (face scutes visible) | Species + **individual ID** |
| `right` | **Right side of the head, close-up** (face scutes visible) | Species + **individual ID** |
| `front` | Head from the front | Species |
| `shell` | Shell from above | Species |
| `rear` | Rear | Species |

**`left` and `right` are what identify an individual green turtle.** Without at least one sharp
face close-up, a green turtle gets species only (`identification.status = "insufficient_views"`).

Optional header **`Idempotency-Key: <your submission id>`**. If a request is retried (timeout,
double tap), sending the same key for the same `uid` returns the original job instead of running
the models twice. Recommended.

Response `200` (success):

```json
{
  "success": true,
  "job_id": "3b1e...",
  "uid": "user-42",
  "species": "green_turtle",
  "species_confidence": 0.94,
  "turtle_id": "GT-000042",
  "status": "matched",
  "confidence": 0.91,
  "error": null
}
```

- `species`: one of the `species.label` values below.
- `turtle_id`, `status`, `confidence`: the individual identification (`status` values are listed
  under `identification.status` below). `turtle_id` and `confidence` are `null` for `skipped` and
  `insufficient_views`.
- For per-photo details (which photo was rejected and why), call `GET /v1/jobs/{job_id}`.

Failure responses:

```json
502 { "success": false, "job_id": "3b1e...", "uid": "user-42", "error": "processing failed" }
504 { "success": false, "job_id": "3b1e...", "uid": "user-42", "error": "still processing, try GET /v1/jobs/3b1e..." }
```

## 2. List a user's submissions

`GET /v1/users/{uid}/jobs?limit=20`

Returns `{ "items": [ <job>, ... ] }`, newest first, same shape as below.
For the next page, pass `before=<created_at of the last item>`.

## 3. Get one job

`GET /v1/jobs/{job_id}`

`status`: `queued` → `running` → `completed` | `failed`

Completed example:

```json
{
  "job_id": "3b1e...",
  "uid": "user-42",
  "status": "completed",
  "error": null,
  "created_at": "2026-10-17T05:10:00Z",
  "updated_at": "2026-10-17T05:10:07Z",
  "result": {
    "species": {
      "label": "green_turtle",
      "confidence": 0.94,
      "scores": { "green_turtle": 0.94, "hawksbill": 0.03, "loggerhead": 0.01, "olive_ridley": 0.01,
                  "leatherback": 0.0, "kemps_ridley": 0.0, "flatback": 0.01 }
    },
    "identification": {
      "status": "matched",
      "turtle_id": "GT-000042",
      "confidence": 0.91,
      "is_provisional": false
    },
    "images": [
      { "index": 1, "slot": "left",  "view": "left_face", "usable": true,  "reason": null,         "detection_confidence": 0.97 },
      { "index": 3, "slot": "shell", "view": "carapace",  "usable": false, "reason": "too_blurry", "detection_confidence": 0.88 }
    ],
    "model_versions": { "detector": "rfdetr-v1", "species": "bioclip-2.5", "segmenter": "seg-v1", "embedder": "embed-v1" }
  }
}
```

### `species.label`

| Value | Meaning |
|---|---|
| `green_turtle`, `hawksbill`, `loggerhead`, `olive_ridley`, `leatherback`, `kemps_ridley`, `flatback` | Identified species |
| `data_deficient` | Not enough usable photos (fewer than 2 passed the quality checks). Ask the user to retake. |
| `unknown` | Photos were fine but the model is not confident (below 50%). |

### `identification.status`

| Value | `turtle_id` | Show the user |
|---|---|---|
| `matched` | existing ID | "This turtle is already known: GT-000042" |
| `new` | new ID | "New turtle registered: GT-000107" |
| `uncertain` | provisional ID | "Possible match, pending expert review" (`is_provisional: true`) |
| `skipped` | null | Not a green turtle, so no individual ID |
| `insufficient_views` | null | Green turtle, but no usable left/right face photo. Ask for face photos |

### `images[].reason` (when `usable` is false)

`download_failed` (link broken or expired), `no_turtle_detected`, `low_detection_confidence`, `turtle_too_small`, `too_blurry`, `scutes_not_found`
Use `slot` + `reason` to tell the user exactly which photo to retake.

### Failed job

```json
{ "job_id": "...", "status": "failed", "result": null, "error": "..." }
```

Show a generic "Something went wrong, please try again".

---

## Webhook (optional, server-to-server)

If the frontend's backend gives us a URL, every finished job is POSTed to it:

```json
{ "job_id": "3b1e...", "uid": "user-42", "status": "completed", "result": { ...same as above... }, "error": null }
```

On failure: `"status": "failed"`, `"result": null`.

Verify it before trusting it. Header `X-Turtle-Signature` = hex HMAC-SHA256 of the **raw request body**
using the shared secret:

```js
import crypto from "node:crypto";
const expected = crypto.createHmac("sha256", process.env.TURTLE_WEBHOOK_SECRET).update(rawBody).digest("hex");
const ok = crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(req.headers["x-turtle-signature"] ?? ""));
```

Respond 2xx quickly. Failed deliveries are retried 3 times; polling still works if all fail.
Use `uid` to route the result to the right user (push notification, socket, etc.).
Do not call the webhook from the browser: it must be a server endpoint, because it holds the secret.

---

## Other endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check (no key needed) |
| GET | `/ready` | Database reachable (no key needed) |
| GET | `/v1/species` | List of all species labels |
| GET | `/v1/turtles/{turtle_id}` | Turtle record + sighting history. If merged, returns `{ "id", "merged_into" }`, so follow `merged_into` |

## Admin endpoints (header `X-Admin-Key`)

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/v1/admin/review?status=pending` | | Uncertain matches waiting for review, with candidate turtles |
| POST | `/v1/admin/review/{id}/resolve` | `{"action":"confirm_new"}` or `{"action":"merge","target_turtle_id":"GT-000042"}` | Resolve a review item |
| POST | `/v1/admin/turtles/merge` | `{"source_turtle_id":"GT-000107","target_turtle_id":"GT-000042"}` | Merge duplicates |
| POST | `/v1/admin/reindex` | | Rebuild vectors after a model change |

## Errors

| Code | When |
|---|---|
| 401 | Missing or wrong key |
| 404 | Job or turtle not found |
| 413 | Request body over 32 KB |
| 422 | Invalid body (missing `uid`, no images, unknown slot, non-https or disallowed host). `detail` says which field |
| 429 | Rate limit: more than 10 submissions per minute for one `uid`. Wait `Retry-After` seconds |
| 500 | Unexpected error. Body has a `request_id`; send it to us |
| 502 | Processing failed (`success: false`). Ask the user to try again |
| 503 | Model service unreachable, retry later (with the same `Idempotency-Key`) |
| 504 | Still processing after 4 minutes (`success: false`). Get the result later with `GET /v1/jobs/{job_id}` |

Every response carries an `X-Request-ID` header. Include it when reporting a problem.
You may also send your own `X-Request-ID` to trace a request across both systems.
