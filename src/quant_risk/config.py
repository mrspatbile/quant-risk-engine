# src/quant_risk/config.py

"""
Project-wide configuration and environment variable loading.

Import this module at the start of any script or notebook to ensure
all API keys and environment variables are available.

Usage:
    from quant_risk.config import FRED_API_KEY, config
"""

import os
from dotenv import load_dotenv

# load .env file from project root
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
load_dotenv(os.path.join(_project_root, ".env"))

# API keys
FRED_API_KEY = os.getenv("FRED_API_KEY")
BCB_API_KEY  = os.getenv("BCB_API_KEY", None)   # optional

# config dict for convenience
config = {
    "FRED_API_KEY" : FRED_API_KEY,
    "BCB_API_KEY"  : BCB_API_KEY,
    "project_root" : _project_root,
}

if __name__ == "__main__":
    print(f"FRED API key : {'SET' if FRED_API_KEY else 'NOT SET'}")
    print(f"BCB API key  : {'SET' if BCB_API_KEY else 'NOT SET'}")
    print(f"Project root : {_project_root}")