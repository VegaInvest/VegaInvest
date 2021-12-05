from flask import Blueprint, request, session, redirect, url_for, render_template, jsonify

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


portfolio_blueprint = Blueprint('portfolios', __name__)


@portfolio_blueprint.route('/portfolio')
@user_decorators.requires_login
def get_portfolio_page(portfolio_id):   # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    fig = port.plot_portfolio()
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()

    return render_template('/portfolios/portfolio.jinja2', portfolio=port, plot_url=plot_data)


@portfolio_blueprint.route('/editrisk', methods=['GET', 'POST'])
@user_decorators.requires_login
# Views form to change portfolio's associated risk aversion parameter
def change_risk(portfolio_id):
    port = Portfolio.get_by_id(portfolio_id)
    if request.method == "POST":
        risk_appetite = request.form['risk_appetite']
        port.risk_appetite = risk_appetite
        port.save_to_mongo()
        fig = port.runMVO()
        canvas = FigureCanvas(fig)
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('/portfolios/optimal_portfolio.jinja2', portfolio=port, plot_url=plot_data)

    return render_template('/portfolios/edit_portfolio.jinja2', portfolio=port)


@portfolio_blueprint.route('/new', methods=['GET', 'POST'])
# @user_decorators.requires_login
def create_portfolio():            # Views form to create portfolio associated with active/ loggedin user
    if request.method == "POST":
        risk_appetite = request.get_json().get('risk_appetite')
        email = request.get_json().get('email')
        amount_invests=get_json().get('amount_invests')
        goal=get_json().get('goal')
        horizon=get_json().get('horizon')
        port = Portfolio(email, risk_appetite=risk_appetite,amount_invest=amount_invests,goal=goal,horizon=horizon)
        port.save_to_mongo()
        fig = port.runMVO()
        # canvas = FigureCanvas(fig)
        # img = BytesIO()
        # fig.savefig(img)
        # img.seek(0)
        # plot_data = base64.b64encode(img.read()).decode()
        return jsonify({'Status': 'portfolio created!'})
    return jsonify({'Status': 'error use POST request'})


@portfolio_blueprint.route('/pushPortfolioid/<string:email>', methods=['GET', 'POST'])
# @user_decorators.requires_login
# Views form to create portfolio associated with active/ loggedin user
def pushportid(email):
    email = str(email)
    if request.method == "GET":
        port_data = Database.find_one(PortfolioConstants.COLLECTION,
                                      {'user_email': email})
        Portfolio_ID = port_data["_id"]
        Portfolio_ID = str(Portfolio_ID)
        return jsonify({'Portfolio_ID': Portfolio_ID})
    return jsonify({'Status': 'error use POST request'})
