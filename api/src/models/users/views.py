from flask import Blueprint, request, session, redirect, url_for, render_template, jsonify
from src.models.users.user import User
import src.models.users.errors as UserErrors
import src.models.users.decorators as user_decorators

user_blueprint = Blueprint("users", __name__)


@user_blueprint.route("/login", methods=["GET", "POST"])
def login_user():  # Views form required for user login
    if request.method == "POST":
        # email = request.form["email"]
        # password = request.form["password"]
        email = request.get_json().get('email')
        password = request.get_json().get("password")
        try:
            if User.is_login_valid(email, password):
                return(jsonify({'Status': 'found'}))

        except UserErrors.UserError as e:
            e = str(e)
            return(jsonify({'Status': e}))

    return jsonify({'Status': 'error use POST request'})


@user_blueprint.route("/register", methods=["GET", "POST"])
def register_user():  # Views form required for user signup
    if request.method == "POST":
        email = request.get_json().get('email')
        password = request.get_json().get("password")
        try:
            if User.register_user(email, password):
                return (jsonify({'Status': 'registered'}))
        except UserErrors.UserError as e:
            e = str(e)
            return (jsonify({'Status': e}))

    return jsonify({'Status': 'error use POST request'})


@user_blueprint.route("/logout")
def logout_user():  # Logs user out
    session["email"] = None
    return redirect(url_for("home"))


# @user_blueprint.route("/portfolios")
# @user_decorators.requires_login
# def user_portfolios():  # Views list of user portfolios
#     print('1',session["email"])
#     user = User.find_by_email(session["email"])
#     portfolios = user.get_portfolios()

#     return render_template("/users/portfolios.jinja2", portfolios=portfolios)
