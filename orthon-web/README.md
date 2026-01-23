# Ørthon Explorer

**geometry leads — ørthon**

Web-based explorer for Ørthon analysis results.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  BROWSER                                                    │
│  - Editorial-style UI                                       │
│  - Folder picker (GUI)                                      │
│  - DuckDB-WASM for SQL queries                              │
│  - Charts with Plotly                                       │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP
┌─────────────────────────────────────────────────────────────┐
│  FASTAPI BACKEND (Mac/VM)                                   │
│  - Serves folder browser                                    │
│  - Serves parquet files                                     │
│  - Runs pipeline on uploads                                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

Open http://localhost:8000 in your browser.

## Usage

1. **Select folder** — Browse to a folder containing your analysis results
2. **Check files** — Green dots show which expected files are present
3. **Load data** — Click "Load Data" to load parquets into browser
4. **Explore** — Use tabs to explore Vector, Geometry, State, Cohort
5. **SQL** — Run custom queries in the SQL Console tab

## Expected Files

The explorer looks for these parquet files:

| File | Description |
|------|-------------|
| `vector.parquet` | Behavioral metrics per signal |
| `geometry.parquet` | Pairwise relationships |
| `state.parquet` | Coherence tracking, regime detection |
| `cohort.parquet` | Behavioral groupings |
| `derivatives.parquet` | (Optional) Rate of change features |

## Config File

If present, `orthon_config.json` provides window options:

```json
{
  "domain": "cmapss",
  "windows": [252, 126, 63],
  "stride": 21,
  "entity_col": "unit_id",
  "time_col": "cycle"
}
```

## Deployment

### Local (Mac)

```bash
python server.py
# Open http://localhost:8000
```

### Cloud VM

```bash
# On VM
python server.py
# Open http://YOUR_VM_IP:8000
```

Add authentication for production use.

## Files

```
orthon-web/
├── server.py        # FastAPI backend
├── index.html       # Frontend (Editorial-style)
├── requirements.txt # Python dependencies
└── README.md        # This file
```
