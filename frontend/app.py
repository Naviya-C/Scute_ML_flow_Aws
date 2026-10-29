"""Streamlit test client for the Sea Turtle ID API.

For each photo it does what a real frontend would:
  1. POST /v1/uploads            -> presigned S3 slot (key, url, fields)
  2. POST <url> (multipart)      -> the photo goes straight to S3
  3. POST /v1/identify           -> the keys of all photos, waits for the result
The API key is read from frontend/.env and stays on the machine running Streamlit.
"""
import os
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

API_BASE = os.environ.get("API_BASE_URL", "http://16.176.24.116:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
DEFAULT_UID = os.environ.get("DEFAULT_UID", "test-user")
IDENTIFY_TIMEOUT = int(os.environ.get("IDENTIFY_TIMEOUT", "300"))  # GPU cold start can take 1-2 min

REQUIRED = {
    "whole": "Whole turtle",
    "left": "Left side of the head (close-up)",
    "right": "Right side of the head (close-up)",
}
OPTIONAL = {
    "front": "Head from the front",
    "shell": "Shell from above",
    "rear": "Rear",
}
TYPES = ["jpg", "jpeg", "png", "webp"]
CONTENT_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def headers() -> dict:
    return {"X-API-Key": API_KEY}


def content_type(file) -> str:
    ext = file.name.rsplit(".", 1)[-1].lower()
    return CONTENT_TYPES.get(ext, file.type or "image/jpeg")


def upload_photo(file) -> str:
    """Presigned-POST one photo to S3 and return its key."""
    ctype = content_type(file)
    r = requests.post(f"{API_BASE}/v1/uploads", json={"content_type": ctype},
                      headers=headers(), timeout=30)
    r.raise_for_status()
    slot = r.json()
    # S3 needs the signed fields first and the file last; requests keeps that order.
    s3 = requests.post(slot["url"], data=slot["fields"],
                       files={"file": (file.name, file.getvalue(), ctype)}, timeout=120)
    if s3.status_code != 204:
        raise RuntimeError(f"S3 upload failed ({s3.status_code}): {s3.text[:300]}")
    return slot["key"]


def show_error(r: requests.Response) -> None:
    try:
        body = r.json()
    except ValueError:
        body = r.text
    st.error(f"HTTP {r.status_code}")
    st.json(body) if isinstance(body, (dict, list)) else st.code(body)


st.set_page_config(page_title="Sea Turtle ID test client", page_icon="🐢", layout="wide")
st.title("🐢 Sea Turtle ID: test client")

with st.sidebar:
    st.header("Connection")
    st.text(f"API: {API_BASE}")
    if not API_KEY:
        st.error("API_KEY is not set. Put it in frontend/.env")
    if st.button("Check /health and /ready"):
        for path in ("/health", "/ready"):
            try:
                r = requests.get(f"{API_BASE}{path}", timeout=10)
                st.write(f"`{path}` -> {r.status_code}", r.json() if r.ok else r.text)
            except requests.RequestException as e:
                st.error(f"`{path}` failed: {e}")

tab_identify, tab_job = st.tabs(["Identify", "Look up a job"])

with tab_identify:
    uid = st.text_input("uid (your user's id)", value=DEFAULT_UID)

    files = {}
    st.subheader("Required photos")
    cols = st.columns(3)
    for col, (slot, label) in zip(cols, REQUIRED.items()):
        with col:
            files[slot] = st.file_uploader(f"{slot}: {label}", type=TYPES, key=slot)
            if files[slot]:
                st.image(files[slot], use_container_width=True)

    st.subheader("Optional photos")
    cols = st.columns(3)
    for col, (slot, label) in zip(cols, OPTIONAL.items()):
        with col:
            files[slot] = st.file_uploader(f"{slot}: {label}", type=TYPES, key=slot)
            if files[slot]:
                st.image(files[slot], use_container_width=True)

    missing = [s for s in REQUIRED if not files[s]]
    if missing:
        st.info("Still needed: " + ", ".join(missing))

    if st.button("Identify", type="primary", disabled=bool(missing or not uid or not API_KEY)):
        keys = {}
        try:
            with st.status("Uploading photos to S3...", expanded=True) as status:
                for slot, f in files.items():
                    if f:
                        keys[slot] = upload_photo(f)
                        st.write(f"✓ {slot} -> `{keys[slot]}`")
                status.update(label="Uploaded. Waiting for the models (first run can take 1-2 min)...")
                r = requests.post(
                    f"{API_BASE}/v1/identify",
                    json={"uid": uid, "images": keys},
                    headers={**headers(), "Idempotency-Key": uuid.uuid4().hex},
                    timeout=IDENTIFY_TIMEOUT,
                )
                status.update(label=f"Done (HTTP {r.status_code})", state="complete")
        except (requests.RequestException, RuntimeError) as e:
            st.error(str(e))
            st.stop()

        if not r.ok:
            show_error(r)
            if r.status_code == 504 and r.headers.get("content-type", "").startswith("application/json"):
                st.session_state["job_id"] = r.json().get("job_id", "")
        else:
            res = r.json()
            st.session_state["job_id"] = res.get("job_id", "")
            if res.get("status") in ("matched", "new"):
                st.success(f"{res['status'].upper()}: {res.get('turtle_id')}")
            elif res.get("status") == "uncertain":
                st.warning(f"Possible match, pending review: {res.get('turtle_id')}")
            elif res.get("status") == "skipped":
                st.info("Leatherback: species recorded, no individual ID (no scutes).")
            else:
                st.error(f"{res.get('status')}: {res.get('reason')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Species", res.get("species") or "-")
            c2.metric("Species confidence", res.get("species_confidence") or "-")
            c3.metric("Match confidence", res.get("confidence") or "-")
            with st.expander("Raw response"):
                st.json(res)

with tab_job:
    job_id = st.text_input("job_id", value=st.session_state.get("job_id", ""))
    if st.button("Get job", disabled=not job_id):
        try:
            r = requests.get(f"{API_BASE}/v1/jobs/{job_id}", headers=headers(), timeout=30)
        except requests.RequestException as e:
            st.error(str(e))
        else:
            if r.ok:
                job = r.json()
                st.write(f"Status: **{job['status']}**")
                for img in (job.get("result") or {}).get("images", []):
                    ok = "✅" if img.get("usable") else "❌"
                    st.write(f"{ok} {img.get('slot')}: {img.get('reason') or 'ok'} "
                             f"(turtle score {img.get('turtle_score')})")
                st.json(job)
            else:
                show_error(r)
