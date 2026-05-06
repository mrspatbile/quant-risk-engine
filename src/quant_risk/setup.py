# src/quant_risk/setup.py

def base():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option("display.float_format", "{:.6f}".format)

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

    print("base loaded")
    return np, pd, plt


def asset_pricing():
    import QuantLib as ql
    from quant_risk.config import PROCESSED_DIR
    from quant_risk.curves.ois import OISCurve
    from quant_risk.curves.nss import NSSCurve

    ois_curve = OISCurve.from_processed(str(PROCESSED_DIR))
    nss_curve = NSSCurve.from_ecb(rating="AAA")

    print(ois_curve.describe())
    print(nss_curve.describe())

    val_parts      = ois_curve.valuation_date.split("-")
    valuation_date = ql.Date(int(val_parts[2]), int(val_parts[1]), int(val_parts[0]))
    ql.Settings.instance().evaluationDate = valuation_date
    calendar       = ql.TARGET()

    print(f"Valuation date : {valuation_date}")
    print(f"Calendar       : TARGET")
    print("asset pricing loaded")

    return ois_curve, nss_curve, valuation_date, calendar, ql


def macro():
    from quant_risk.config import FRED_API_KEY
    from quant_risk.data.fed import FedClient
    from quant_risk.data.fed_store import FREDStore
    from quant_risk.data.external_store import ExternalStore

    if FRED_API_KEY is None:
        raise ValueError("FRED_API_KEY missing in config")

    fed_client     = FedClient(api_key=FRED_API_KEY)
    store          = FREDStore(FRED_API_KEY)
    external_store = ExternalStore()

    print("macro loaded")
    return fed_client, store, external_store