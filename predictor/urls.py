from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("analytics/", views.analytics, name="analytics"),
    path("history/", views.history, name="history"),

    # Authentication
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # Admin dashboard
    path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("admin-dashboard/users/", views.manage_users, name="manage_users"),
    path(
        "admin-dashboard/users/<int:user_id>/toggle-status/",
        views.toggle_user_status,
        name="toggle_user_status",
    ),
    path(
    "admin-dashboard/predictions/",
    views.all_predictions,
    name="all_predictions",
    ),
]