"""FastAPI application for serving the urgency model.

Endpoints:
    GET  /health   -> {"status": "ok", "model": "<backbone>", "run": "<run_name>"}
    POST /predict  -> multipart/form-data with 'file' field -> InferenceResult JSON

All responses include a disclaimer field.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from urgency.inference.infer import UrgencyInferer
from urgency.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level inferer, initialized once at startup
_inferer: UrgencyInferer | None = None
_run_dir: Path | None = None


def create_app(run_dir: Path) -> FastAPI:
    """Create the FastAPI application for a specific run directory.

    Args:
        run_dir: Path to a completed training run directory.

    Returns:
        Configured FastAPI application.
    """
    global _run_dir
    _run_dir = Path(run_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[misc]
        global _inferer
        logger.info("Loading model from %s ...", _run_dir)
        _inferer = UrgencyInferer(_run_dir)
        logger.info("Model loaded. API ready.")
        yield
        _inferer = None

    app = FastAPI(
        title="ISIC Lesion Urgency API",
        description=(
            "Research tool for predicting skin lesion urgency from images. "
            "NOT a diagnostic device. Not for clinical use."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        if _inferer is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        return {
            "status": "ok",
            "model": _inferer.cfg.model.backbone,
            "run": _inferer.cfg.run_name,
            "disclaimer": (
                "This is a research tool, not a diagnostic device. "
                "For research use only."
            ),
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        """Predict lesion urgency from an uploaded image.

        Accepts JPEG or PNG images. Returns urgency probabilities and
        a triage decision.
        """
        if _inferer is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        allowed_types = {"image/jpeg", "image/png", "image/jpg"}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. "
                       f"Accepted types: {allowed_types}",
            )

        # Write to a temp file so PIL can read it
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = _inferer.predict(tmp_path)
        except Exception as exc:
            logger.exception("Inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        return JSONResponse(content=result.to_dict())

    return app
