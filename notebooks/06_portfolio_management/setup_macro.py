"""
Shared setup for macro / FRED-based notebooks.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt


from quant_risk.data.fed import FedClient
from quant_risk.config import FRED_API_KEY

pd.set_option("display.float_format", "{:.6f}".format)

# ---------------------------------------------------------------------
# FRED client
# ---------------------------------------------------------------------

if FRED_API_KEY is None:
    raise ValueError("FRED_API_KEY is missing in config")

fed_client = FedClient(api_key=FRED_API_KEY)

print("FedClient initialized")



# =========================
# STYLE
# =========================

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
})  