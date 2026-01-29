"""
PRISM API Routes
================

HTTP endpoints for manifest-based compute.

ORTHON pushes:
  - observations.parquet → data/input/
  - manifest.json → data/input/

PRISM runs ManifestRunner, writes results to data/output/
"""

from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time
import json
import shutil
from datetime import datetime

from prism.runner import ManifestRunner

# Data directories
DATA_DIR = Path(__file__).parent.parent.parent / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PRISM",
    description="Manifest-based compute for signal primitives",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/compute")
async def compute(
    observations: UploadFile = File(...),
    manifest: UploadFile = File(...),
):
    """
    Manifest-based compute endpoint.

    Receives:
        - observations: observations.parquet file
        - manifest: manifest.json file

    Returns:
        - status: complete/error
        - output_dir: path to results
        - files: list of output parquet files
    """
    start = time.time()

    # Create job directory
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_input = INPUT_DIR / job_id
    job_output = OUTPUT_DIR / job_id
    job_input.mkdir(parents=True, exist_ok=True)
    job_output.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded files
        obs_path = job_input / "observations.parquet"
        manifest_path = job_input / "manifest.json"

        obs_content = await observations.read()
        with open(obs_path, 'wb') as f:
            f.write(obs_content)

        manifest_content = await manifest.read()
        manifest_dict = json.loads(manifest_content)

        # Inject paths into manifest
        manifest_dict['observations_path'] = str(obs_path)
        manifest_dict['output_dir'] = str(job_output)

        # Save manifest with paths
        with open(manifest_path, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

        # Run ManifestRunner
        runner = ManifestRunner(manifest_dict)
        result = runner.run()

        duration = time.time() - start

        # List output files
        output_files = [f.name for f in job_output.glob("*.parquet")]

        return {
            "status": "complete",
            "job_id": job_id,
            "duration_seconds": round(duration, 2),
            "output_dir": str(job_output),
            "files": output_files,
            "file_urls": [f"/results/{job_id}/{f}" for f in output_files],
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "job_id": job_id,
                "message": str(e),
            }
        )


@app.get("/results/{job_id}/{filename}")
async def get_result(job_id: str, filename: str):
    """Download a result parquet file."""
    # Security: prevent path traversal
    import os
    safe_job_id = os.path.basename(job_id)
    safe_filename = os.path.basename(filename)

    file_path = OUTPUT_DIR / safe_job_id / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=file_path,
        filename=safe_filename,
        media_type="application/octet-stream"
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = []
    for job_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if job_dir.is_dir():
            files = list(job_dir.glob("*.parquet"))
            jobs.append({
                "job_id": job_dir.name,
                "files": [f.name for f in files],
                "file_count": len(files),
            })
    return {"jobs": jobs[:20]}  # Last 20 jobs


@app.get("/engines")
async def list_engines():
    """List available engines."""
    from prism.python_runner import SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
    from prism.sql_runner import SQL_ENGINES

    return {
        "signal": SIGNAL_ENGINES,
        "pair": PAIR_ENGINES,
        "symmetric_pair": SYMMETRIC_PAIR_ENGINES,
        "windowed": WINDOWED_ENGINES,
        "sql": SQL_ENGINES,
        "total": len(SIGNAL_ENGINES) + len(PAIR_ENGINES) + len(SYMMETRIC_PAIR_ENGINES) + len(WINDOWED_ENGINES) + len(SQL_ENGINES)
    }
