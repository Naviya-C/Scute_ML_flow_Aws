"""Admin endpoints (header X-Admin-Key): review queue, merges, re-index."""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .. import dispatch
from ..db import get_db
from ..models import Embedding, ReviewItem, Sighting, Turtle
from ..schemas import MergeIn, ResolveIn
from ..security import require_admin

router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin)])


def merge(db: Session, source: str, target: str) -> None:
    """Fold `source` into `target`: vectors, sightings and earlier merges all move over."""
    if source == target:
        raise HTTPException(400, "source and target are the same")
    tgt, src = db.get(Turtle, target), db.get(Turtle, source)
    if not tgt or not src:
        raise HTTPException(404, "turtle not found")
    if tgt.merged_into:
        raise HTTPException(400, f"target is merged into {tgt.merged_into}")
    now = func.now()
    db.execute(update(Embedding).where(Embedding.turtle_id == source).values(turtle_id=target))
    db.execute(update(Sighting).where(Sighting.turtle_id == source)
               .values(turtle_id=target, match_status="matched"))
    db.execute(update(Turtle).where(Turtle.merged_into == source)
               .values(merged_into=target, updated_at=now))
    db.execute(update(Turtle).where(Turtle.id == source).values(merged_into=target, updated_at=now))
    db.execute(update(ReviewItem)
               .where(ReviewItem.provisional_turtle_id == source, ReviewItem.status == "pending")
               .values(status="resolved", resolved_at=now,
                       resolution={"action": "merge", "target_turtle_id": target}))


@router.get("/review")
def list_review(status: str = "pending", limit: int = 50, db: Session = Depends(get_db)):
    rows = db.execute(
        select(ReviewItem, Sighting.job_id).join(Sighting, Sighting.id == ReviewItem.sighting_id)
        .where(ReviewItem.status == status).order_by(ReviewItem.created_at)
        .limit(max(1, min(limit, 200)))
    ).all()
    return {"items": [{
        "id": str(r.id), "job_id": str(job_id), "sighting_id": str(r.sighting_id),
        "provisional_turtle_id": r.provisional_turtle_id, "candidates": r.candidates,
        "status": r.status, "resolution": r.resolution, "created_at": r.created_at,
    } for r, job_id in rows]}


@router.post("/review/{review_id}/resolve")
def resolve_review(review_id: uuid.UUID, body: ResolveIn, db: Session = Depends(get_db)):
    r = db.scalar(select(ReviewItem).where(ReviewItem.id == review_id).with_for_update())
    if not r or r.status != "pending":
        raise HTTPException(404, "pending review item not found")
    prov = r.provisional_turtle_id
    if body.action == "confirm_new":
        db.execute(update(Turtle).where(Turtle.id == prov)
                   .values(status="confirmed", updated_at=func.now()))
        db.execute(update(Sighting).where(Sighting.id == r.sighting_id).values(match_status="new"))
        r.status, r.resolved_at, r.resolution = "resolved", func.now(), {"action": "confirm_new"}
        final_id = prov
    else:
        if not body.target_turtle_id:
            raise HTTPException(400, "target_turtle_id required for merge")
        merge(db, prov, body.target_turtle_id)
        final_id = body.target_turtle_id
    db.commit()
    return {"review_id": str(review_id), "turtle_id": final_id}


@router.post("/turtles/merge")
def merge_turtles(body: MergeIn, db: Session = Depends(get_db)):
    merge(db, body.source_turtle_id, body.target_turtle_id)
    db.commit()
    return {"merged": body.source_turtle_id, "into": body.target_turtle_id}


@router.post("/reindex", status_code=202)
def reindex():
    return {"call_id": dispatch.start_reindex(),
            "note": "runs in background; bump model versions in the Modal secret first"}
