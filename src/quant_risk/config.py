import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

# paths
PROJECT_ROOT  = _project_root
CACHE_DIR     = PROJECT_ROOT / "data" / "cache"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# API keys
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
BCB_API_KEY  = os.getenv("BCB_API_KEY", "").strip() or None

config = {
    "FRED_API_KEY"  : FRED_API_KEY,
    "BCB_API_KEY"   : BCB_API_KEY,
    "project_root"  : PROJECT_ROOT,
    "cache_dir"     : CACHE_DIR,
    "processed_dir" : PROCESSED_DIR,
}

if __name__ == "__main__":
    print(f"FRED API key : {'SET' if FRED_API_KEY else 'NOT SET'}")
    print(f"BCB API key  : {'SET' if BCB_API_KEY else 'NOT SET'}")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Cache dir    : {CACHE_DIR}")