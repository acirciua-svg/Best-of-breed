# BEST-OF-BREED — Fundamental Screener

Reads `data/fundamentals.csv`, validates the schema, and produces market-cap–weighted outputs.

## Quick start
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r requirements.txt
python screen.py --input data/fundamentals.csv --outdir outputs
