import datetime

COLLECTION = "portfolios"  # MongoDB collection
TICKERS = [
    "QCLN",
    "SOXX",
    "PTF",
    "LIT",
    "PTH",
    "FINX",
    "IPO",
    "PSJ",
    "IWY",
    "KCE",
    "XWEB",
    "FDIS",
    "GAMR",
    "PALL",
    "IAI",
    "IHI",
    "IMCG",
    "ITB",
    "PSCH",
    "RTH",
    "SOCL",
    "BOTZ",
    "PTNQ",
    "BBC",
    "ERTH",
]  # List of tickers used
# TICKERS =['AAPL','AMZN','MMM','T','KO']
# Gamma values corresponding to risk appetites
RISK_PROFILE_INDECES = [32, 39, 60]
RISK_LABELS = ["high", "medium", "low"]  # Different portfolio risk levels
RISK_APP_DICT = dict(
    zip(RISK_LABELS, RISK_PROFILE_INDECES)
)  # Dict tying gamma values to risk levels
# very first start date for our etf universe
START_DATE = datetime.datetime(2016, 9, 14)
# end date for simulation will be present
END_DATE = datetime.datetime(2021, 1, 2)
# END_DATE = datetime.datetime.now() can change to datetime.datetime.now()
