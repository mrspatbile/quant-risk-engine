"""
Shared setup for asset pricing notebooks.
Run this cell in any notebook to load curves and set display options.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
import os

from quant_risk.curves.ois import OISCurve
from quant_risk.curves.nss import NSSCurve

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", "{:.6f}".format)

project_root  = os.path.abspath(os.path.join(os.getcwd(), "../.."))
processed_dir = os.path.join(project_root, "data", "processed")

ois_curve = OISCurve.from_processed(processed_dir)
nss_curve = NSSCurve.from_ecb(rating="AAA")

print(ois_curve.describe())
print(nss_curve.describe())