import uuid
import yfinance as yf

# from models.stocks.constants import COLLAPSE
from src.common.database import Database
import src.models.stocks.constants as StockConstants
import src.models.stocks.errors as StockErrors
from alpha_vantage.timeseries import TimeSeries
import time
import datetime
import pandas as pd
import json
import numpy as np

ts = TimeSeries(key=StockConstants.API, output_format=StockConstants.OUTPUTFORMAT)


# def get_rawprices(ticker, collapse):
#     if collapse=="monthly":
#         data, meta_data = ts.get_monthly_adjusted(ticker)
#     if collapse=="daily":
#         data, meta_data = ts.get_daily_adjusted(ticker)
#     return data


class Stock(object):
    def __init__(self, ticker, returns, mu, std, _id=None):
        # Stock class creates stock instances of assets stored/allowed
        # Only needs to enter ticker name and run get_Params to fill in the rest.
        self.ticker = ticker
        self.returns = returns
        self.mu = mu
        self.std = std
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Asset: {}>".format(self.ticker)

    def get_rawprices(ticker, collapse):
        if collapse == "monthly":
            data, meta_data = ts.get_monthly_adjusted(ticker)
        if collapse == "daily":
            data, meta_data = ts.get_daily_adjusted(ticker)
        return data

    def get_fromAV(ticker, mindate, maxdate):
        data = Stock.get_rawprices(ticker, "daily")
        data = data[["4. close"]]
        data.rename(columns={"4. close": ticker}, inplace=True)
        # print(data)
        data = data[mindate:maxdate]
        return data

    def update_rawData():
        date = datetime.datetime.today()
        # add if date.hour==22: update record everyday do this
        date = datetime.datetime(date.year, date.month, date.day, 0, 0)
        data = yf.download(StockConstants.TICKERS, start=date, end=date, interval="1d")[
            "Adj Close"
        ]
        data.columns.name = ""
        data.reset_index(inplace=True)
        data.rename(columns={"index": "Ticker"}, inplace=True)
        data.columns = data.columns.astype(str)
        data = data.dropna()
        Database.update("rawdata", {}, {"$set": data.to_dict("record")[0]})

    def push_rawData(start_date, end_date):
        data = yf.download(
            StockConstants.TICKERS, start=start_date, end=end_date, interval="1d"
        )["Adj Close"]
        data.columns.name = ""

        count = 0
        # loop through tickers
        for ticker in data.columns:  # col
            NA = False
            init = datetime.datetime(1000, 1, 1, 0, 0)
            mindate = init
            maxdate = init
            # loop through dates
            for date in data.index:
                # determine the min and max dates for ticker with NA price
                if pd.isna(data.loc[date, ticker]):
                    NA = True
                    if mindate == init:
                        mindate = date.strftime("%Y/%m/%d")
                    if maxdate < date:
                        maxdate = date
            # replace the price with alphavantage
            if maxdate != init or mindate != init:
                if count % 5 == 0 and count != 0:
                    time.sleep(63)
                av_data = Stock.get_fromAV(ticker, mindate, maxdate)
                count += 1
                if len(av_data) != 0:
                    for ticker in av_data.columns:
                        for date in av_data.index:
                            data.loc[date, ticker] = av_data.loc[date, ticker]
                print(count, " 1st loop ", maxdate, ":", mindate, " ticker:", ticker)
            elif mindate != init and NA:
                if count % 5 == 0 and count != 0:
                    time.sleep(63)
                av_data = Stock.get_fromAV(ticker, mindate, mindate)
                count += 1
                if len(av_data) != 0:
                    for ticker in av_data.columns:
                        for date in av_data.index:
                            data.loc[date, ticker] = av_data.loc[date, ticker]
                print(count, " 2nd loop ", maxdate, ":", mindate), " ticker:", ticker
        #         print("Ticker:",ticker,mindate,":",maxdate)

        #    print(maxdate,mindate)

        data.reset_index(inplace=True)
        data.dropna(inplace=False)
        data.rename(columns={"index": "Ticker"}, inplace=True)
        data.columns = data.columns.astype(str)
        Database.initialize()
        Database.update("rawdata", {}, {"$set": data.to_dict("record")[0]}, upsert=True)

    def get_from_db(startdate, enddate):
        # Stock.push_rawData(startdate,enddate)
        # TO pull data from mongodb

        results = Database.find("rawdata", {})
        # print(list(blah)[:])
        datab = pd.DataFrame.from_dict(
            results, orient="columns", dtype=None, columns=None
        )
        datab.drop(columns="_id", inplace=True)
        datab.set_index("Date", inplace=True)
        return datab

    @classmethod
    def get_Params(cls, ticker, start_date, end_date):
        """
        Gets ticker data from Quandl API and saves stock to database

        :param ticker: {type:string} Asset Ticker (ex: 'AAPL')
        :param start_date: {type:string} time-series start date (ex: YYYY-MM-DD '2006-01-01')
        :param end_date: {type:string} time-series end date (ex: YYYY-MM-DD '2006-01-01')
        :return: Stock instance
        """

        error = False
        # try:
        data = Stock.get_from_db(start_date, end_date).iloc[::-1]
        data = data[[ticker]]
        data.columns = [ticker]

        # except:
        error = True

        # if error is True:
        # raise StockErrors.IncorrectTickerError("The ticker {} is invalid!".format(ticker))

        rets = data.pct_change().dropna()

        mu = rets.mean().values[0]
        std = rets.std().values[0]

        stock = cls(
            ticker=ticker, returns=rets.to_json(orient="index"), mu=mu, std=std
        )  # create instance of stock
        stock.save_to_mongo()  # save instance to db

        return stock

    def save_to_mongo(self):
        Database.update(StockConstants.COLLECTION, {"_id": self._id}, self.json())

    def json(self):  # Creates JSON representation of stock instance
        return {
            "_id": self._id,
            "ticker": self.ticker,
            "returns": self.returns,
            "mu": self.mu,
            "std": self.std,
        }

    def check(ticker):
        return Database.fine_one()

    @classmethod
    def get_by_id(cls, stock_id):  # Retrieves stock from MongoDB by its unique id
        return cls(**Database.find_one(StockConstants.COLLECTION, {"_id": stock_id}))

    @classmethod
    def get_by_ticker(
        cls, stock_ticker
    ):  # Retrieves stock from MongoDB by its unique ticker
        return cls(
            **Database.find_one(StockConstants.COLLECTION, {"ticker": stock_ticker})
        )

    @classmethod
    def all(cls):  # Retrieves all stock records in MongoDB
        return [cls(**elem) for elem in Database.find(StockConstants.COLLECTION, {})]

    def remove(self):  # Removes stock from MongoDB by its unique id
        return Database.remove(StockConstants.COLLECTION, {"_id": self._id})
