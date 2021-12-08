from flask import (
    Blueprint,
    request,
    session,
    redirect,
    url_for,
    render_template,
    jsonify,
)
import json
from pandas.core.tools.datetimes import to_datetime
from dateutil.relativedelta import relativedelta
from src.models.users.user import User
import src.models.users.errors as UserErrors
import src.models.users.decorators as user_decorators
from src.common.database import Database
from src.models.portfolios.portfolio import Portfolio
import src.models.portfolios.constants as PortfolioConstants
import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import urllib
import base64
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


portfolio_blueprint = Blueprint("portfolios", __name__)


@portfolio_blueprint.route("/portfolio")
@user_decorators.requires_login
def get_portfolio_page(portfolio_id):  # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    # fig = port.plot_portfolio()
    # canvas = FigureCanvas(fig)
    # img = BytesIO()
    # fig.savefig(img)
    # img.seek(0)
    # plot_data = base64.b64encode(img.read()).decode()

    return render_template(
        "/portfolios/portfolio.jinja2", portfolio=port, plot_url=plot_data
    )


@portfolio_blueprint.route("/editrisk", methods=["GET", "POST"])
@user_decorators.requires_login
# Views form to change portfolio's associated risk aversion parameter
def change_risk(portfolio_id):
    port = Portfolio.get_by_id(portfolio_id)
    if request.method == "POST":
        risk_appetite = request.form["risk_appetite"]
        port.risk_appetite = risk_appetite
        port.save_to_mongo()
        fig = port.runMVO()
        canvas = FigureCanvas(fig)
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template(
            "/portfolios/optimal_portfolio.jinja2", portfolio=port, plot_url=plot_data
        )

    return render_template("/portfolios/edit_portfolio.jinja2", portfolio=port)


@portfolio_blueprint.route("/new", methods=["GET", "POST"])
# @user_decorators.requires_login
def create_portfolio():  # Views form to create portfolio associated with active/ loggedin user
    if request.method == "POST":
        risk_appetite = request.get_json().get("risk_appetite")
        email = request.get_json().get("email")
        ermsg = ""
        e = 0
        c = 0
        amount_invest = request.get_json().get("amount_invest")
        goal = request.get_json().get("goal")
        horizon = request.get_json().get("horizon")

        if not str.isnumeric(amount_invest):
            if e == 0:
                msg = "Amount Invested is not a valid type"
                ermsg = msg
            e += 1
        if not str.isnumeric(goal):
            if e == 0:
                msg = "Goal is not a valid type"
                ermsg = ermsg + msg
            e += 1
        if not str.isnumeric(horizon):
            if e == 0:
                msg = "Horizon is not a valid type"
                ermsg = ermsg + msg
            e += 1
        if str.isnumeric(amount_invest) and float(amount_invest) < 0:
            if e == 0:
                msg = "Amount Invested must be greater than $0!"
                ermsg = ermsg + msg
            e += 1
        if str.isnumeric(goal) and float(goal) < 0:
            if e == 0:
                msg = "Goal must be greater than $0!"
                ermsg = ermsg + msg
            e += 1
        if str.isnumeric(horizon) and float(horizon) < 0:
            if e == 0:
                msg = "Horizon must be greater than $0!"
                ermsg = ermsg + msg
            e += 1
        if (
            str.isnumeric(goal)
            and str.isnumeric(amount_invest)
            and float(goal) < float(amount_invest)
        ):
            if e == 0:
                msg = "Goal must be higher than Amount Invested"
                ermsg = ermsg + msg
            e += 1

        if e > 0:
            return jsonify({"Status": ermsg})
        else:
            amount_invest = float(amount_invest)
            last_updated = PortfolioConstants.END_DATE 
            start = PortfolioConstants.END_DATE 
            goal = float(goal)
            horizon = float(horizon)
            end = int(relativedelta(PortfolioConstants.END_DATE, PortfolioConstants.START_DATE).years)
            outs = Portfolio.multi_period_backtesting(PortfolioConstants.TICKERS, forecast_window=4, lookback=7, estimation_model=linear_model.SGDRegressor(random_state=42, max_iter=5000), alpha=.1, gamma_trans=.1, gamma_risk=100000, date=Portfolio.to_integer(PortfolioConstants.START_DATE), end=end*12, risk_appetite=risk_appetite)
            curr_weights=outs[0][-1]
            ann_returns=outs[1][1]
            ann_vol=outs[1][2]
            sharpe=outs[1][3]
            port_val=outs[1][-1]    
            date_vector = outs[1][-2]
            port = Portfolio(
                email,
                risk_appetite=risk_appetite,
                amount_invest=amount_invest,
                goal=goal,
                horizon=horizon,
                curr_weights=curr_weights.tolist(),
                ann_returns=ann_returns,
                ann_vol=ann_vol,
                sharpe=sharpe,
                port_val=port_val.tolist(),
                last_updated = last_updated,
                start=start,
                date_vector = date_vector.tolist()
            )
            port.save_to_mongo()

            #X[1][1] is annualized returns
            #X[1][2] is vol
            #X[1][3] is sharpe
            #X[1][-1] is vector of Portfolio value

            # canvas = FigureCanvas(fig)
            # img = BytesIO()
            # fig.savefig(img)
            # img.seek(0)
            # plot_data = base64.b64encode(img.read()).decode()
            return jsonify({"Status": "portfolio created!"})
    return jsonify({"Status": "error use POST request"})


# @portfolio_blueprint.route("/pushPortfolioid/<string:email>", methods=["GET", "POST"])
# # @user_decorators.requires_login
# # Views form to create portfolio associated with active/ loggedin user
# def pushportid(email):
#     email = str(email)
#     if request.method == "GET":
#         port_data = Database.find_one(
#             PortfolioConstants.COLLECTION, {"user_email": email}
#         )
#         Portfolio_ID = port_data["_id"]
#         Portfolio_ID = str(Portfolio_ID)
#         return jsonify({"Portfolio_ID": Portfolio_ID})
#     return jsonify({"Status": "error use POST request"})


# @user_decorators.requires_login
# Views form to create portfolio associated with active/ loggedin user
@portfolio_blueprint.route("/pushWeights/<string:email>", methods=["GET", "POST"])
def pushWeights(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )
        date_vector = port_data['date_vector']
        date_vector = date_vector.tolist()
        weights = port_data["curr_weights"]
        weights=np.around(weights, 3)
        weights=weights.tolist()
        #print(weights*100)
        #time_index=time_index.tolist()
        #X[1][1] is annualized returns
        #X[1][2] is vol
        #X[1][3] is sharpe
        #X[1][-1] is vector of Portfolio value
        #X[1][-2] is time vector    
        #date in yyyymmdd format, start at 7 periods (months) before required start date

        return jsonify(
            {
                "Status": "Success",
                "weights": weights,
                "date_vector" : date_vector
            }
        )
    return jsonify({"Status": "error use POST request"})

@portfolio_blueprint.route("/pushParams/<string:email>", methods=["GET", "POST"])
def pushParams(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )

        last_updated = port_data['last_updated']
        start=port_data['start']
        date_vector = port_data['date_vector']
        time_diff_update = int(relativedelta(last_updated, start).months)

        if 
        risk_appetite = port_data["risk_appetite"]
        risk_appetite = str(risk_appetite)
        horizon = port_data["horizon"]
        horizon = float(horizon)
        goal = port_data["goal"]
        goal = float(goal)
        amount_invested = port_data["amount_invest"]
        amount_invested = float(amount_invested)
        ann_returns=float(port_data["ann_returns"])
        ann_vol=float(port_data["ann_vol"])
        sharpe=float(port_data["sharpe"])
        port_val=np.array(port_data["port_val"])
        # x = Portfolio.multi_period_backtesting(PortfolioConstants.TICKERS, forecast_window=4, lookback=7, estimation_model=linear_model.SGDRegressor(random_state=42, max_iter=5000),  alpha=.1, gamma_trans=10, gamma_risk=1000, date=Portfolio.to_integer(PortfolioConstants.START_DATE), end=36, risk_appetite=risk_appetite)
        time_difference = int(relativedelta(PortfolioConstants.END_DATE, PortfolioConstants.START_DATE).years)


        sharpe=np.round(sharpe,3)
        returns=np.round(ann_returns,3)
        vol=np.round(ann_vol,3)
        portval=np.around(port_val*float(amount_invested), 3)
        lastportval =portval[-1]
        portval=portval.tolist()
        #time_index=time_index.tolist()
        #X[1][1] is annualized returns
        #X[1][2] is vol
        #X[1][3] is sharpe
        #X[1][-1] is vector of Portfolio value
        #date in yyyymmdd format, start at 7 periods (months) before required start date

        return jsonify(
            {
                "Status": "Success",
                "risk_appetite": risk_appetite,
                "horizon": horizon,
                "goal": goal,
                "amount_invested": amount_invested,
                'sharpe' : sharpe,
                'returns' : returns,
                'vol' : vol,
                'time_difference' : time_difference,
                'portval' : portval,
                'lastportval' : lastportval,
            }
        )
    return jsonify({"Status": "error use POST request"})

# @portfolio_blueprint.route("/pushWeights/<string:email>", methods=["GET", "POST"])
# def pushParams(email):
#     email = str(email)
#     if request.method == "POST":
#         port_data = Database.find_one(
#             PortfolioConstants.COLLECTION, {"user_email": email}
#         )
#         risk_appetite = port_data["risk_appetite"]
#         risk_appetite = str(risk_appetite)
#         horizon = port_data["horizon"]
#         horizon = float(horizon)
#         goal = port_data["goal"]
#         goal = float(goal)
#         print(risk_appetite)
#         return jsonify(
#             {
#                 "Status": "Success",
#                 "risk_appetite": risk_appetite,
#                 "horizon": horizon,
#                 "goal": goal,
#             }
#         )
#     return jsonify({"Status": "error use POST request"})

# @portfolio_blueprint.route("/pushWeights/<string:email>", methods=["GET", "POST"])
# def pushParams(email):
#     email = str(email)
#     if request.method == "POST":
#         port_data = Database.find_one(
#             PortfolioConstants.COLLECTION, {"user_email": email}
#         )
#         risk_appetite = port_data["risk_appetite"]
#         risk_appetite = str(risk_appetite)
#         horizon = port_data["horizon"]
#         horizon = float(horizon)
#         goal = port_data["goal"]
#         goal = float(goal)
#         print(risk_appetite)
#         return jsonify(
#             {
#                 "Status": "Success",
#                 "risk_appetite": risk_appetite,
#                 "horizon": horizon,
#                 "goal": goal,
#             }
#         )
 #   return jsonify({"Status": "error use POST request"})