from pandas.core.indexes import datetimes
from src.models.portfolios.constants import START_DATE
from src.models.stocks.stock import Stock
import src.models.portfolios.constants as PortfolioConstants
from src.common.database import Database
import cvxpy as cvx
import datetime
import uuid
import numpy as np
import pandas as pd
import matplotlib
import datetime
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

plt.style.use("ggplot")
import cvxpy as cvx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from numpy import matlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.style.use("ggplot")


class Portfolio(object):
    # Portfolio class creates portfolio instances for user portfolios using stocks in Stock class
    # def __init__(self, user_email, risk_appetite, tickers=None, weights=None, _id=None):
    def __init__(
        self,
        user_email,
        risk_appetite,
        amount_invest,
        goal,
        horizon,
        curr_weights,
        ann_returns,
        ann_vol,
        sharpe,
        port_val,
        last_updated,
        start,
        date_vector,
        tickers=None,
        _id=None,
    ):
        self.user_email = user_email
        self.risk_appetite = risk_appetite
        self.tickers = PortfolioConstants.TICKERS if tickers is None else tickers
        self.amount_invest = amount_invest
        self.goal = goal
        self.horizon = horizon
        self.curr_weights = curr_weights
        self.ann_returns = ann_returns
        self.ann_vol = ann_vol
        self.sharpe = sharpe
        self.port_val = port_val
        self.last_updated = last_updated
        self.start = start
        self.date_vector = date_vector
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<Portfolio for user {}>".format(self.user_email)

    # def get_Params(self,start_date=PortfolioConstants.START_DATE,end_date=PortfolioConstants.END_DATE,tickers=PortfolioConstants.TICKERS,lookback):
    def get_Params(
        self, start_date, end_date, tickers, lookback, forecast_window, estimation_model
    ):
        data = Portfolio.Import_data_inputs(start_date, tickers)

        excess_ret = data[0].resample("M").agg(lambda x: (x + 1).prod() - 1)
        factor_ret = data[1].resample("M").agg(lambda x: (x + 1).prod() - 1)
        raw_rets = data[2].resample("M").agg(lambda x: (x + 1).prod() - 1)

        n_stocks = len(tickers)

        params = Portfolio.Param_forecast(
            np.array(excess_ret[-lookback:]),
            np.array(factor_ret[-len(excess_ret) :])[-lookback:],
            lookback=7,
            forecast=forecast_window,
            model=estimation_model,
        )
        mu = params[0].transpose()
        Q = params[1]
        return mu, Q, raw_rets, n_stocks

    # def get_Params(
    #     self,
    #     start_date=PortfolioConstants.START_DATE,
    #     end_date=PortfolioConstants.END_DATE,
    # ):
    #     """
    #     Checks MongoDB stocks collection to see if assets already exist. If not, runs Stock.get_Params to
    #     get asset return time series from Quandl. Once all time series are collected, computes vector of
    #     expected returns and the covariance matrix

    #     :param start_date: time-series start date as string format (ex: YYYY-MM-DD '2006-01-01')
    #     :param end_date: time-series start date as string format (ex: YYYY-MM-DD '2016-12-31')
    #     :param tickers: list of tickers to retrieve

    #     :return: Expected Return, Covariance Matrix, Variance, Standard Deviations
    #     """

    #     tickers = self.tickers.copy()

    #     n = len(tickers)
    #     rets = []
    #     mu = []
    #     if Portfolio.check_collection("rawdata") == False:
    #         Stock.push_rawData(start_date, end_date)

    #     for ticker in tickers:
    #         try:
    #             stock = Stock.get_by_ticker(
    #                 stock_ticker=ticker
    #             )  # Check if stock exists in db
    #         except:  # If not, get params from Quandl
    #             stock = Stock.get_Params(
    #                 ticker=ticker, start_date=start_date, end_date=end_date
    #             )

    #         rets.append(pd.read_json(stock.returns, orient="index"))
    #         mu.append(stock.mu)
    #         returns = pd.concat(rets, axis=1)

    #     mu = np.array(mu).reshape([n, 1])
    #     cov = returns.cov()
    #     cov = cov.values
    #     np.fill_diagonal(cov, 0.1)
    #     return mu, cov
    def multi_period_backtesting(tickers, forecast_window, lookback, estimation_model, alpha, gamma_trans, gamma_risk, date, end, risk_appetite):

        date = max(20160914,date)
        #print("Start Date: ", date)
        #print(date,"\n")
        
        if risk_appetite == "high":
            print("Multi Period Sharpe Ratio Optimization")
        elif risk_appetite == "low":
            print("Multi Period Risk Parity Optimization")
        else:
            print("Multi Period MVO")

        data = Portfolio.Import_data_inputs(date, tickers)

        excess_ret = data[0].resample('M').agg(lambda x: (x + 1).prod() - 1)
        factor_ret = data[1].resample('M').agg(lambda x: (x + 1).prod() - 1)
        raw_rets = data[2].resample('M').agg(lambda x: (x + 1).prod() - 1)
        print(raw_rets[["PALL"]])
        weights = []

        height = (len(excess_ret)-lookback)//forecast_window
        n_stocks = len(tickers)
        dates = excess_ret.index[-lookback:]
        for i in range(height):
            
            params = Portfolio.Param_forecast(np.array(excess_ret[i*forecast_window:i*forecast_window+lookback]), np.array(factor_ret[-len(excess_ret):])[i*forecast_window:i*forecast_window+lookback], lookback=7, forecast=forecast_window, model=estimation_model)
            mu = params[0].transpose()
            Q = params[1]
            rbt_mu = Portfolio.robust_mu(mu, Q, alpha, forecast_window)
            if risk_appetite == 'high':
                weights = weights + [np.array(Portfolio.multi_sharpe(mu, Q, forecast_window, gamma_trans, gamma_risk))]
            elif risk_appetite == 'low':
                weights = weights + [np.array(Portfolio.multi_rp(rbt_mu, Q, forecast_window, gamma_trans, gamma_risk))]
            else:
                weights = weights + [np.array(Portfolio.multi_period_mvo(rbt_mu, Q, forecast_window, gamma_trans, gamma_risk))]


        weights = np.array(weights).reshape(height*forecast_window,len(tickers))

        weights = np.array(weights)
        
        raw_rets = np.array(raw_rets)[-len(weights):]
        bench = np.array(factor_ret.iloc[:,0])[-len(weights):]
        rfr = np.array(factor_ret.iloc[:,5])[-len(weights):]
        dates=(excess_ret.index.to_numpy())[-len(weights)-1:]

        if end == 0:
            results = Portfolio.single_period_portfolio_backtest(raw_rets, weights, bench, rfr, dates)

        else:
            results = Portfolio.single_period_portfolio_backtest(raw_rets[:end], weights[:end], bench[:end], rfr[:end], dates[:end+1])
        print('\n')
        return weights, results
        #X[1][1] is annualized returns
        #X[1][2] is vol
        #X[1][3] is sharpe
        #X[1][-1] is vector of Portfolio value
        #X[1][-2] is time vector
        #date in yyyymmdd format, start at 7 periods (months) before required start date

    def Param_forecast(input_stock_rets, input_factor_rets, lookback, forecast, model):
        # forecast mu and Q based on lookback and selected regression model
        if forecast > lookback:
            print("Warning! Increase lookback length to display full forecast.")
        forecast = min(forecast, lookback)

        input_stock_rets = np.array(input_stock_rets)
        input_factor_rets = np.array(input_factor_rets)

        num_assets = input_stock_rets.shape[1]
        F = Portfolio.factor_forecast(input_factor_rets, lookback, forecast)
        mu = []

        for i in range(num_assets):
            mu = mu + [
                Portfolio.beta_forecast(
                    input_factor_rets,
                    F,
                    input_stock_rets.transpose()[i],
                    lookback,
                    forecast,
                    model,
                )[2]
            ]

        mu = np.array(mu)
        Q = Portfolio.cov_forecast(input_stock_rets, mu.transpose(), lookback, forecast)

        return mu, Q

    def factor_forecast(factor_rets, lookback, forecast):
        # forecast factors based on simple geometric means
        # ARIMA was tested but worked too inconsistently to be used
        output = np.full((forecast, factor_rets.shape[1]), np.nan)
        rolling = factor_rets[-lookback:]

        for i in range(forecast):
            # output[i] = np.array(pd.DataFrame(rolling).mean())
            output[i] = np.array(gmean(pd.DataFrame(rolling) + 1) - 1)
            rolling = np.vstack([rolling[1:], output[i].transpose()])
        return output

    def beta_forecast(
        historical_factor_rets,
        factor_forecast,
        single_stock_ret,
        lookback,
        forecast,
        model,
    ):  
        # single stock forecast of factor betas throughout time consistent with given lookback for given forecast period
        betas = np.full((forecast, factor_forecast.shape[1]), np.nan)
        alphas = np.full(forecast, np.nan)
        rolling = single_stock_ret[-lookback:]
        factors = np.vstack([historical_factor_rets[-lookback:], factor_forecast])

        for i in range(forecast):
            x = model.fit(factors[i : i + lookback], rolling)
            betas[i] = x.coef_
            alphas[i] = x.intercept_
            ret = np.matmul(betas[i], factor_forecast[i]) + alphas[i]
            rolling = np.append(rolling[1:], [ret])
            mu = rolling[lookback - forecast :]

        return alphas, betas, mu

    def cov_forecast(rets_historical, rets_forecast, lookback, forecast): 
        #creates cov matrix of predictions consistent with lookbac period
        Q = []
        rets = np.vstack([rets_historical[-lookback:], rets_forecast])

        for i in range(forecast):
            Q = Q + [np.cov(rets[i : i + lookback].transpose())]

        Q = np.array(Q)

        return Q

    def robust_mu(mu, Q, alpha, forecast_window): 
        # creates mu consistent with long-only robust optimzation
        robust_mu = []
        for i in range(forecast_window):
            robust_mu = robust_mu + [mu[i] - alpha * (np.diag(Q[i])**.5)]
        robust_mu = np.array(robust_mu)
        return robust_mu

    def to_integer(dt_time):
        return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day

    def backend_output(
        self,
        tickers,
        forecast_window,
        lookback,
        estimation_model,
        alpha,
        gamma_trans,
        gamma_risk,
        risk_appetite,
    ):

        date = Portfolio.to_integer(PortfolioConstants.START_DATE)

        if risk_appetite == "high":
            print("Multi Period Sharpe Ratio Optimization")
        elif risk_appetite == "low":
            print("Multi Period Risk Parity Optimization")
        else:
            print("Multi Period MVO")
        # def get_Params(self,start_date=PortfolioConstants.START_DATE,end_date=PortfolioConstants.END_DATE,tickers=PortfolioConstants.TICKERS,lookback):
        # def get_Params(self,start_date,end_date,tickers,lookback, forecast_window,estimation_model):

        mu, Q, raw_rets, n_stocks = Portfolio.get_Params(
            self,
            start_date=Portfolio.to_integer(PortfolioConstants.START_DATE),
            end_date=Portfolio.to_integer(PortfolioConstants.END_DATE),
            tickers=PortfolioConstants.TICKERS,
            lookback=lookback,
            forecast_window=forecast_window,
            estimation_model=estimation_model,
        )

        rbt_mu = Portfolio.robust_mu(mu, Q, alpha, forecast_window)
        if risk_appetite == "high":
            weights = np.array(
                Portfolio.multi_sharpe(
                    rbt_mu, Q, forecast_window, gamma_trans, gamma_risk
                )
            )
        elif risk_appetite == "low":
            weights = np.array(
                Portfolio.multi_rp(rbt_mu, Q, forecast_window, gamma_trans, gamma_risk)
            )
        else:
            #medium
            weights = np.array(
                Portfolio.multi_period_mvo(
                    rbt_mu, Q, forecast_window, gamma_trans, gamma_risk
                )
            )

        weights = np.array(weights)
        # print(self.weights.shape)
         #self.save_to_mongo()
        Portfolio.weights_to_df(self, weights, tickers)

        return weights

    


    def single_period_portfolio_backtest(rets, weights, benchmark, rfr, dates): 
        #returns portfolio performance metrics

        portf_ret = np.diag(np.matmul(rets, weights.transpose()))
        benchmark = benchmark + rfr

        Annual = (gmean(portf_ret - rfr + 1)) ** 12 - 1
        Vol = np.std(portf_ret) * (12 ** 0.5)
        Sharpe = (gmean(portf_ret - rfr + 1)**12 - 1) / (np.std(portf_ret - rfr)*(12**0.5))
        Information = (gmean((portf_ret - benchmark) + 1) - 1) / np.std(
            portf_ret - benchmark
        )
        Sortino = (gmean(portf_ret - rfr + 1) - 1) / np.std(portf_ret[portf_ret >= 0])
        Beta = np.cov(portf_ret, benchmark)[0, 1] / np.var(benchmark)

        cum_returns = (1 + pd.DataFrame(portf_ret)).cumprod()
        drawdown = 1 - cum_returns.div(cum_returns.cummax())

        # print(
        #     "\n",
        #     "Avg Annual Returns: ",
        #     Annual,
        #     "\n",
        #     "Avg Annual Volatility: ",
        #     Vol,
        #     "\n",
        #     "Sharpe Ratio: ",
        #     Sharpe,
        #     "\n",
        #     "Information Ratio: ",
        #     Information,
        #     "\n",
        #     "Sortino Ratio: ",
        #     Sortino,
        #     "\n",
        #     "Benchmark Beta: ",
        #     Beta,
        #     "\n",
        #     "Max Drawdown: ",
        #     drawdown.max()[0],
        #     "\n",
        #     "\n",
        # )

        t = dates

        portf_ret = pd.DataFrame(portf_ret + 1).cumprod()
        benchmark = pd.DataFrame(
            benchmark + np.array(rfr)[-len(weights) :] + 1
        ).cumprod()

        portf_ret = np.insert(np.array(portf_ret), 0, 1)
        benchmark = np.insert(np.array(benchmark), 0, 1)

        # plt.plot(t, portf_ret, "g-", t, benchmark, "--")
        # plt.show()
        # #print("Green - Portfolio Returns")

        return (
            portf_ret[-1], 
            Annual,
            Vol,
            Sharpe,
            Information,
            Sortino,
            drawdown.max()[0], t, portf_ret
        )
        #1. final return multiple
        #2. annual ret
        #3. annual vol
        #4. annual sharpe
        #5/6. Information/Sortino ratios
        #7. max drawdown
        #8. dates
        #9. porfolio value vector (multiplier)

    def multi_period_mvo(mu, cov, forecast, gamma_trans, gamma_risk):
        n = len(mu[0])
        w = np.full(len(mu[0]), 0)

        prob_arr = []
        z_vars = []

        beebee = []
        for tau in range(forecast):
            z = cvx.Variable(*w.shape)

            wplus = w + z

            obj = Portfolio.cost_function(
                mu, tau, wplus, gamma_risk, z, gamma_trans, cov
            )

            constr = []

            constr += [cvx.sum(wplus) == 1]
            constr += [wplus >= 0]
            constr += [wplus <= 1 / 3]

            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(wplus)
            w = wplus

        test = sum(prob_arr).solve(solver=cvx.SCS)

        for i in z_vars:
            beebee.append(i.value)
        return beebee

    def cost_function(mu, tau, wplus, gamma_risk, z, gamma_trans, cov):

        exp_ret = mu[tau].T * wplus
        risk = cvx.quad_form(wplus, cov[tau])

        return exp_ret - gamma_risk * risk - sum(cvx.abs(z) * gamma_trans)

    def multi_rp(mu, cov, forecast, gamma_trans, gamma_risk):
        n = len(mu[0])
        w = np.full(len(mu[0]), 0)
        prob_arr = []
        z_vars = []
        beebee = []

        for tau in range(forecast):
            y = cvx.Variable(n)
            wplus = w + y
            constr = []
            y_sum = 0
            for j in range(n):
                y_sum += cvx.log(y[j])

            obj = (
                0.5 * cvx.quad_form(y, cov[tau]) - y_sum + gamma_trans * sum(cvx.abs(y))
            )

            constr += [wplus >= 0]

            prob = cvx.Problem(cvx.Minimize(obj), constr)
            prob_arr.append(prob)
            w = wplus
            weights = wplus / sum(wplus)
            z_vars.append(weights)

        test = sum(prob_arr).solve(solver=cvx.SCS)
        for i in z_vars:
            beebee.append(i.value)
        return beebee

    def multi_sharpe(mu, cov, forecast, gamma_trans, gamma_risk):
        rf = 0.0025 # fed funds rate upper bound to add robustness to model, 
        # can be changed to actual rfr data, but max since 2016 is 21.02 bps
        rf_hat = np.ones(len(mu[0])) * rf
        one_vec = np.ones(len(mu[0]))
        w = np.full(len(mu[0]), 0)
        prob_arr = []
        z_vars = []
        beebee = []
        # gamma_trans = 30
        # gamma_risk = 1
        for tau in range(forecast):
            y = cvx.Variable(*rf_hat.shape)
            z = cvx.Variable(*rf_hat.shape)
            r_excess = mu[tau] - rf_hat

            wplus = w + z

            kappa = (np.power(r_excess, -1)).T * wplus
            obj = cvx.quad_form(y, cov[tau]) + gamma_trans * sum(cvx.abs(z))

            constr = []
            constr += [cvx.sum(wplus) == 1]
            constr += [wplus >= 0]
            constr += [r_excess.T * y == 1]
            constr += [cvx.sum(y) == kappa]
            constr += [kappa >= 0]
            constr += [y >= 0]

            prob = cvx.Problem(cvx.Minimize(obj), constr)
            prob_arr.append(prob)
            weights = y / cvx.sum(y)
            z_vars.append(weights)
            w = wplus

        test = sum(prob_arr).solve(solver=cvx.ECOS)

        for i in z_vars:
            beebee.append(i.value)
        return beebee

    # def runMVO(
    #     self,
    #     start_date=PortfolioConstants.START_DATE,
    #     end_date=PortfolioConstants.END_DATE,
    #     samples=100,
    # ):
    #     """
    #     (A) Gets Portfolio characteristics (exp. return & cov. matrix)
    #     (B) Runs basic MVO for different risk aversion values (gamma)
    #     (C) Portfolio weights given current portfolio's risk aversion are extracted and assigned to
    #         current portfolio instance which is then updated in MongoDB
    #     (D) A figure of the efficient frontier for all risk aversion parameters is created along with
    #         the individual assets used.

    #     :param start_date: time-series start date as string format (ex: YYYY-MM-DD '2006-01-01')
    #     :param end_date: time-series start date as string format (ex: YYYY-MM-DD '2016-12-31')
    #     :param samples: number of portfolios to compute for comparison/ efficient frontier

    #     :return: Matplotlib figure object of efficient frontier
    #     """

    #     #        (A)        #
    #     #####################
    #     mu, cov = self.get_Params(start_date, end_date)
    #     std = np.sqrt(np.diag(cov))
    #     n = len(mu)

    #     #        (B)        #
    #     #####################
    #     w = cvx.Variable(n)
    #     gamma = cvx.Parameter(nonneg=True)
    #     exp_ret = mu.T * w
    #     cov[cov < 0] = 0
    #     risk = cvx.quad_form(w, cov)

    #     prob = cvx.Problem(
    #         cvx.Maximize(exp_ret - gamma * risk), [cvx.sum(w) == 1, w >= 0]
    #     )

    #     SAMPLES = samples
    #     risk_data = np.zeros(SAMPLES)
    #     ret_data = np.zeros(SAMPLES)
    #     ws = []
    #     gamma_vals = np.logspace(-2, 4, num=SAMPLES)

    #     for i in range(SAMPLES):
    #         gamma.value = gamma_vals[i]
    #         prob.solve()
    #         risk_data[i] = cvx.sqrt(risk).value
    #         ws.append(np.array(w.value).T.tolist())
    #         ret_data[i] = exp_ret.value

    #     w_minVar = cvx.Variable(n)
    #     exp_ret_minVar = mu.T * w_minVar

    #     risk_minVar = cvx.quad_form(w_minVar, cov)
    #     prob_minVar = cvx.Problem(
    #         cvx.Minimize(risk_minVar), [cvx.sum(w_minVar) == 1, w_minVar >= 0]
    #     )
    #     prob_minVar.solve()
    #     risk_data_minVar = cvx.sqrt(risk_minVar).value
    #     ret_data_minVar = exp_ret_minVar.value

    #     #        (C)        #
    #     #####################
    #     gam = PortfolioConstants.RISK_APP_DICT.get(self.risk_appetite)

    #     port_weights = ws[gam]
    #     self.weights = port_weights
    #     self.save_to_mongo()

    #     #        (D)        #
    #     #####################
    #     fig = self.plot_comparison(
    #         risk_data, ret_data, gamma_vals, risk_data_minVar, ret_data_minVar, std, mu
    #     )

    #     return fig

    def Import_data_inputs(startdate, tickers):
        startdate = max(20160914, startdate)

        index = len(tickers)

        stocks_rets = Portfolio.Import_stocks(startdate, tickers)
        factor_rets = Portfolio.Import_factors(startdate)

        merged = pd.merge(
            left=stocks_rets,
            left_index=True,
            right=factor_rets,
            right_index=True,
            how="inner",
        )

        stocks_rets = merged.iloc[:, :index]
        factor_rets = merged.iloc[:, index:]

        rfr = factor_rets.iloc[:, 5]
        excess_rets = stocks_rets.subtract(rfr, axis=0)

        return excess_rets, factor_rets, stocks_rets

    def Import_factors(startdate): #imports Fama-French 5 factors and momentum factor from Kenneth French's website
        url = urlopen(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        )

        # Download FFdata
        zipfile = ZipFile(BytesIO(url.read()))
        FFdata = pd.read_csv(
            zipfile.open("F-F_Research_Data_5_Factors_2x3_daily.CSV"),
            header=0,
            names=["Date", "MKT-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            skiprows=3,
        )
        FFdata = FFdata.loc[FFdata["Date"] >= startdate].set_index("Date")

        # Download momentum
        url = urlopen(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        )

        # Download Zipfile and create pandas DataFrame
        zipfile = ZipFile(BytesIO(url.read()))
        Momdata = pd.read_csv(
            zipfile.open("F-F_Momentum_Factor_daily.CSV"),
            header=0,
            names=["Date", "Mom"],
            skiprows=13,
        )[:-1]
        Momdata["Date"] = Momdata["Date"].astype(int)
        Momdata = Momdata.loc[Momdata["Date"] >= startdate].set_index("Date")

        FFdata = FFdata.join(Momdata) / 100
        FFdata.index = pd.to_datetime(FFdata.index, format="%Y%m%d")

        return FFdata

    def Import_stocks(startdate, tickers): #imports stock adjusted close data from database and converts it to a monthly return
        startdate = startdate - 1
        startdate = (
            str(startdate)[4:6] + "/" + str(startdate)[6:8] + "/" + str(startdate)[0:4]
        )
        startdate = datetime.datetime.strptime(startdate, "%m/%d/%Y")
        end_date = PortfolioConstants.END_DATE #datetime.datetime.today()
        stock_ret = Stock.get_from_db()[tickers]
        stock_ret = (
            stock_ret / stock_ret.shift(1) - 1
        )  # convert prices to daily returns
        stock_ret = stock_ret[1:]
        return stock_ret


    def check_collection(collectionname):
        temp = Database.getCollectionList()
        if collectionname in temp:
            return True
        else:
            return False


    def save_to_mongo(self):
        Database.update(PortfolioConstants.COLLECTION, {"_id": self._id}, self.json())

    def json(self):  # Creates JSON representation of portfolio instance
        return {
            "_id": self._id,
            "user_email": self.user_email,
            "risk_appetite": self.risk_appetite,
            "tickers": self.tickers,
            "amount_invest": self.amount_invest,
            "goal": self.goal,
            "horizon": self.horizon,
            "start": self.start,
            "curr_weights" : self.curr_weights,
            "ann_returns" :  self.ann_returns,
            "ann_vol" :  self.ann_vol,
            "sharpe" :  self.sharpe,
            "port_val" :  self.port_val,
            'last_updated' : self.last_updated,
            'start' : self.start,
            'date_vector' : self.date_vector
        }

    def weights_to_df(self, weights, tickers):
        periods = int(weights.shape[0]) - 1
        weights = pd.DataFrame(
            data=weights, index=list(range(weights.shape[0])), columns=tickers
        )
        weights.reset_index(inplace=True)
        weights.rename(columns={"index": "Period"}, inplace=True)

        for item in weights.to_dict("record"):
            Database.update(
                "portfolioweights",
                {
                    "email": self.user_email,
                    "Period": item["Period"],
                    "start": self.start,
                },
                {"$set": item},
            )
        return True

    def return_PortfolioID(self, email):
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )
        Portfolio_ID = port_data["_id"]
        return Portfolio_ID

    @classmethod
    def get_by_id(cls, port_id):  # Retrieves portfolio from MongoDB by its unique id
        return cls(**Database.find_one(PortfolioConstants.COLLECTION, {"_id": port_id}))

    @classmethod
    def get_by_email(cls, email):  # Retrieves portfolio(s) from MongoDB by user's email
        return [
            cls(**elem)
            for elem in Database.find(
                PortfolioConstants.COLLECTION, {"user_email": email}
            )
        ]