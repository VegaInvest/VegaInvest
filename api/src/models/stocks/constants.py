COLLECTION = "stocks"  # MongoDB collection
API = "RKCOHU52G40NATNE"  # Alpha Vantage API key
OUTPUTFORMAT = "pandas"  # Output format for Alpha Vantage
CLOSECOLUMN = "4. close"
VOLUMECOLUMN = "5. volume"  # Quandl table
COLLAPSE = "monthly"
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
 ]  # List of tickers used         # Quandl data aggregation value
# DESCRIPTIONS=[{'ticker' : 'SOXX', 'description' : 'The iShares Semiconductor ETF tracks the investment results of an index composed of U.S.listed equities in the semiconductor sector.'}, 
#                 { 'ticker' : 'PTF' , 'description' :'Comprised of stocks of various companies based in the technology sector of the market. Invests all of its assets in domestic securities, and holds some big names in the tech sector, i.e., Apple and IBM'},
#                 { 'ticker' : 'LIT' , 'description' :'Invests in the full lithium cycle, from mining and refining the metal,through battery production. Seeks to provide investment results that correspond generally to the price and yield performance of the Solactive Global Lithium Index.'},
#                 { 'ticker' : 'PTH' , 'description' :'Tracks the Dorsey Wright Healthcare Technical Leaders Index.'},
#                 { 'ticker' : 'FINX' , 'description' :'Seeks to invest in companies on the leading edge of the emerging financian technology sector,including industries like insurance, investing, and fundraising. '},
#                 { 'ticker' : 'IPO' , 'description' :'Tracks the Renaissance IPO Index designed to hold a portfolio of the largest, most liquid, newly-listed U.S. IPOs. '},
#                  { 'ticker' : 'PSJ' , 'description' :'Seeks to replicate a benchmark that is comprised of various software companies. Primarily invests in medium cap companies. '},
#                   { 'ticker' : 'IWY' , 'description' :'Provides exposure to large U.S. companies whose earnings are expected to grow at an above-average rate relative to the market.'},
#                    { 'ticker' : 'KCE' , 'description' :'Seeks to provide exposure to the capital markets segment of the S&P TMI, which comprises the following sub-industries: Asset Management & Custody Banks, Diversified Capital Markets, Financial Exchanges & Data, and Investment Banking & Brokerage'},
#                     { 'ticker' : 'XWEB' , 'description' :'Tracks the the S&P Internet Select Industry Index, which includes sub-industies such as Internet & Direct Marketing Retail, Internet Services & Infrastructure'},
#                      { 'ticker' : 'FDIS' , 'description' :'Tracks the Fidelity MSCI Consumer Discretionary Index ETFoffering targeted exposure to the U.S. consumer discretionary sector, i.e., hotel operators, cruise line companies, etc. '},
#                       { 'ticker' : 'PALL' , 'description' :'First in the space that focuses on palladium exposure in an ETF. The fund tracks the movements in palladium spot price.'},
#                        { 'ticker' : 'IAI' , 'description' :'Tracks the investment results of the Dow Jones U.S. Select Investment Services Index, composed of U.S. investment services sector equities. '},
#                         { 'ticker' : 'IHI' , 'description' :'Tracks an index of medical devices U.S. equitiesproviding exposure to U.S. companies that manufacture and distribute medical devices'},
#                         { 'ticker' : 'ITB' , 'description' :'Exposure to U.S. companies that manufacture residential homes, providing targeted accessto domestic home construction stocks'},
#                         { 'ticker' : 'PSCH' , 'description' :'Based on the S&P SmallCap600 Capped Health Care Index. Provides exposure to common stocks in the health care sector, including health care related products, biotechnology, and pharmaceuticals'},
#                         { 'ticker' : 'RTH' , 'description' :'RTH tracks a market-cap-weighted index of the 25 largest US-listed companies that derive most of their revenue from retail.'},
#                         { 'ticker' : 'SOCL' , 'description' :'Tracks the Solactive Social Media Total Return Index, featuring companies that provide social networking, file sharing, and other web-based media. '},
#                         { 'ticker' : 'BOTZ' , 'description' :'Seeks to invest in companies that potentially stand to benefit from increased adoption and utilization of robotics and artificial intelligence (AI), including those involved with industrial robotics and automation, non-industrial robots, and autonomous vehicles.'},
#                         { 'ticker' : 'PTNQ' , 'description' :'PTNQ tracks an index that holds the NASDAQ-100 securities and/or 3-month US T-bills according to momentum.'}
#                         {'ticker': "BBC", 'description':"Tracks the LifeSci Biotechnology Clinical Trials Index, which tracks the performance of select clinical trials stage biotechnology companies."},
#                         {'ticker': "GAMR", 'description':"Tracks an equity index of global firms that support, create or use video games. Stocks are assigned to pure-play, non-pure-play or conglomerate baskets, and weighted equally within each."},
#                         {'ticker': "QLNC", 'description':"Tracks the First Trust NASDAQ Clean Edge Green Energy (alternative energy) Index."},
#                         {'ticker': "SPY", 'description':"Tracks the S&P500 stock market index of large and mid-cap US stocks."},
#                         {'ticker': "ERTH", 'description':"Fund that seeks to track the investment results of MSCI Global Environment Select Index. The Fund generally will invest at least 90% of its total assets in the securities that comprise the Underlying Index. The Underlying Index is designed to maximize exposure to six themes that impact the environment."}]