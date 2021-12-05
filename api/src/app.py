from flask import Flask, render_template, send_from_directory, request
from src.common.database import Database
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS  # comment this on deployment

# Initialize Flask app
app = Flask(__name__)
app.config.from_pyfile("config.py")
# app.config.from_object('config')
app.secret_key = "123"
CORS(app)  # comment this on deployment
api = Api(app)

# Initialize Database before running any other command
@app.before_first_request
def init_db():
    Database.initialize()


# Render home page
@app.route("/")
def home():
    return render_template("home.jinja2")


# @app.route("/yo", methods=["GET", "POST"])
# def test():
#     data = request.get_json()
#     print(data)
#     return '<p> yo </p>'


# Import all views
from src.models.users.views import user_blueprint
from src.models.portfolios.views import portfolio_blueprint
from src.models.stocks.views import stock_blueprint

# Register views in Flask app
app.register_blueprint(user_blueprint, url_prefix="/users")
app.register_blueprint(portfolio_blueprint, url_prefix="/portfolios")
app.register_blueprint(stock_blueprint, url_prefix="/stocks")
