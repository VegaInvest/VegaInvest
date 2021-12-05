from flask import (
    Blueprint,
    request,
    session,
    redirect,
    url_for,
    render_template,
    jsonify,
)

from src.models.users.user import User
import src.models.users.errors as UserErrors
import src.models.users.decorators as user_decorators
from src.common.database import Database
from src.models.portfolios.portfolio import Portfolio
import src.models.portfolios.constants as PortfolioConstants

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import urllib
import base64


portfolio_blueprint = Blueprint("portfolios", __name__)


@portfolio_blueprint.route("/portfolio")
@user_decorators.requires_login
def get_portfolio_page(portfolio_id):  # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    fig = port.plot_portfolio()
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()

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
            goal = float(goal)
            horizon = float(horizon)
            port = Portfolio(
                email,
                risk_appetite=risk_appetite,
                amount_invest=amount_invest,
                goal=goal,
                horizon=horizon,
            )
            port.save_to_mongo()
            fig = port.runMVO()
            # canvas = FigureCanvas(fig)
            # img = BytesIO()
            # fig.savefig(img)
            # img.seek(0)
            # plot_data = base64.b64encode(img.read()).decode()
            return jsonify({"Status": "portfolio created!"})
    return jsonify({"Status": "error use POST request"})


@portfolio_blueprint.route("/pushPortfolioid/<string:email>", methods=["GET", "POST"])
# @user_decorators.requires_login
# Views form to create portfolio associated with active/ loggedin user
def pushportid(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )
        Portfolio_ID = port_data["_id"]
        Portfolio_ID = str(Portfolio_ID)
        return jsonify({"Portfolio_ID": Portfolio_ID})
    return jsonify({"Status": "error use POST request"})


# @user_decorators.requires_login
# Views form to create portfolio associated with active/ loggedin user
@portfolio_blueprint.route("/pushRisk/<string:email>", methods=["GET", "POST"])
def pushportrisk(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )
        risk_appetite = port_data["risk_appetite"]
        risk_appetite = str(risk_appetite)
        print(risk_appetite)
        return jsonify({"Status": "Success", "risk_appetite": risk_appetite})
    return jsonify({"Status": "error use POST request"})


@portfolio_blueprint.route("/pushHorizon/<string:email>", methods=["GET", "POST"])
def pushporthorizon(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )
        horizon = port_data["horizon"]
        horizon = str(horizon)
        return jsonify({"Status": "Success", "horizon": horizon})
    return jsonify({"Status": "error use POST request"})


@portfolio_blueprint.route("/pushGoal/<string:email>", methods=["GET", "POST"])
def pushportgoal(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(
            PortfolioConstants.COLLECTION, {"user_email": email}
        )
        goal = port_data["goal"]
        goal = str(goal)
        return jsonify({"Status": "Success", "goal": goal})
    return jsonify({"Status": "error use POST request"})
