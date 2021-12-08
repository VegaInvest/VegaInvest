# from functools import wraps
# from src.app import app
# from flask import session, redirect, url_for, request


# def requires_login(func):       # Ensures user is logged-in in order to run function
#     @wraps(func)
#     def decorated_function(*args, **kwargs):
#         if session['email'] is None:
#             # if requires login, then redirects to login page, then back to original page
#             # after user is logged in.
#             return redirect(url_for('users.login_user', next=request.path))
#         return func(*args, **kwargs)
#     return decorated_function


# # Ensures logged-in user is an admin in order to run function
# def requires_admin_permissions(func):
#     @wraps(func)
#     def decorated_function(*args, **kwargs):
#         if 'email' not in session.keys() or session['email'] is None:
#             return redirect(url_for('users.login_user', next=request.path))
#         if (session['email'] not in app.config['ADMINS']):
#             return redirect(url_for('users.login_user'))
#         return func(*args, **kwargs)
#     return decorated_function
