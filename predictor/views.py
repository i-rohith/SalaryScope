
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.db.models import Avg
from django.views.decorators.http import require_POST

import json
from datetime import timedelta

from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone


from .models import PredictionHistory

logger = logging.getLogger(__name__)

# Load the trained salary prediction model
MODEL_PATH = (
    Path(settings.BASE_DIR)
    / "models"
    / "salary_prediction_pipeline.pkl"
)

salary_pipeline = joblib.load(MODEL_PATH)


def home(request):
    return render(request, "home.html")


def predict(request):
    context = {
        "page_title": "Salary Prediction",
    }

    if request.method == "POST":
        try:
            # Collect the submitted job details
            input_data = {
                "Job_Title": request.POST.get("job_title", "").strip(),
                "Company": request.POST.get("company", "Unknown").strip()
                or "Unknown",
                "Company_Type": request.POST.get("company_type", "").strip(),
                "Industry": request.POST.get("industry", "").strip(),
                "City": request.POST.get("city", "").strip(),
                "Location_Tier": request.POST.get("location_tier", "").strip(),
                "Experience_Level": request.POST.get("experience_level", "").strip(),
                "Job_Type": request.POST.get("job_type", "").strip(),
                "Work_Mode": request.POST.get("work_mode", "").strip(),
                "Skills_Required": request.POST.get("skills", "").strip(),
                "Education_Required": request.POST.get("education", "").strip(),
            }

            # Check required fields
            if not all(input_data[key] for key in salary_pipeline["features"]):
                context["error"] = "Please fill in all job details."
                return render(request, "predict.html", context)

            # Arrange columns in the exact training order
            input_df = pd.DataFrame(
                [input_data],
                columns=salary_pipeline["features"],
            )

            # Apply the saved preprocessing and prediction model
            processed_data = salary_pipeline["preprocessor"].transform(input_df)
            predicted_log_salary = salary_pipeline["model"].predict(processed_data)

            # Convert the prediction back from log scale to LPA
            predicted_salary = float(np.expm1(predicted_log_salary[0]))

            context["predicted_salary"] = round(predicted_salary, 2)

            # Calculate the 90% estimated salary range
            margin = salary_pipeline.get("calibration_margin_lpa")

            if margin is not None:
                context["lower_salary"] = round(max(0, predicted_salary - margin), 2)
                context["upper_salary"] = round(predicted_salary + margin, 2)
                context["interval_level"] = int(
                    salary_pipeline.get("calibration_level", 0.90) * 100
                )

            context["input_data"] = input_data
            # Prepare prediction details for guest browser history
            context["guest_prediction"] = {
                "job_title": input_data["Job_Title"],
                "company": input_data["Company"],
                "city": input_data["City"],
                "experience_level": input_data["Experience_Level"],
                "predicted_salary": round(predicted_salary, 2),
                "lower_salary": context.get("lower_salary"),
                "upper_salary": context.get("upper_salary"),
                "interval_level": context.get("interval_level", 90),
            }
            if request.user.is_authenticated:
                PredictionHistory.objects.create(
                    user=request.user,
                    job_title=input_data["Job_Title"],
                    company=input_data["Company"],
                    company_type=input_data["Company_Type"],
                    industry=input_data["Industry"],
                    city=input_data["City"],
                    location_tier=input_data["Location_Tier"],
                    experience_level=input_data["Experience_Level"],
                    job_type=input_data["Job_Type"],
                    work_mode=input_data["Work_Mode"],
                    skills_required=input_data["Skills_Required"],
                    education_required=input_data["Education_Required"],
                    predicted_salary=predicted_salary,
                    lower_salary=context.get("lower_salary"),
                    upper_salary=context.get("upper_salary"),
                    interval_level=context.get("interval_level", 90),
                )

        except Exception as error:
            context["error"] = (
                "Unable to predict salary. Please check the submitted details "
                "and try again."
            )
            print("Salary prediction error:", error)

    return render(request, "predict.html", context)


def analytics(request):
    import re

    dataset_path = Path(settings.BASE_DIR) / "india_job_market_2024_2026.csv"

    job_title = request.GET.get("job_title", "").strip()
    selected_experience = request.GET.get("experience_level", "").strip()
    predicted_salary_raw = request.GET.get("predicted_salary", "").strip()

    # Analytics must be associated with a searched job role.
    if not job_title:
        return render(
            request,
            "analytics.html",
            {
                "error_message": (
                    "Please make a salary prediction first, then open "
                    "Job Market Analytics from the prediction result."
                ),
                "job_title": "",
            },
        )

    try:
        df = pd.read_csv(dataset_path)

        required_columns = [
            "Job_Title",
            "Salary_LPA",
            "City",
            "Experience_Level",
            "Openings",
            "Applicants",
            "Skills_Required",
            "Work_Mode",
            "Job_Type",
        ]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing dataset columns: " + ", ".join(missing_columns)
            )

        # Clean relevant fields.
        df["Salary_LPA"] = pd.to_numeric(
            df["Salary_LPA"], errors="coerce"
        )
        df["Openings"] = pd.to_numeric(
            df["Openings"], errors="coerce"
        ).fillna(0).clip(lower=0)
        df["Applicants"] = pd.to_numeric(
            df["Applicants"], errors="coerce"
        ).fillna(0).clip(lower=0)

        df = df.dropna(subset=["Job_Title", "Salary_LPA"])
        df = df[df["Salary_LPA"] >= 0]

        # Match the selected role, ignoring case and surrounding spaces.
        normalized_titles = (
            df["Job_Title"].astype(str).str.strip().str.casefold()
        )
        requested_title = job_title.casefold()

        role_df = df[normalized_titles == requested_title].copy()

        # If no exact title exists, try a partial match.
        if role_df.empty:
            role_df = df[
                normalized_titles.str.contains(
                    re.escape(requested_title), na=False
                )
            ].copy()

        if role_df.empty:
            return render(
                request,
                "analytics.html",
                {
                    "job_title": job_title,
                    "error_message": (
                        f"No matching job records were found for "
                        f"'{job_title}' in the supplied dataset."
                    ),
                },
            )

        # Add unique record IDs for tracking duplicate skills safely
        role_df = role_df.reset_index(drop=True)
        role_df["_record_id"] = role_df.index

        # Salary benchmark for the selected role.
        salary_values = role_df["Salary_LPA"]

        average_salary = float(salary_values.mean())
        median_salary = float(salary_values.median())
        minimum_salary = float(salary_values.min())
        maximum_salary = float(salary_values.max())
        total_records = len(role_df)

        predicted_salary = None
        if predicted_salary_raw:
            try:
                predicted_salary = float(predicted_salary_raw)
                if not np.isfinite(predicted_salary) or predicted_salary < 0:
                    predicted_salary = None
            except (TypeError, ValueError):
                predicted_salary = None

        salary_difference = None
        salary_difference_percent = None

        if predicted_salary is not None:
            salary_difference = predicted_salary - average_salary
            if average_salary != 0:
                salary_difference_percent = (
                    salary_difference / average_salary
                ) * 100

        # Optional benchmark for the user's selected experience.
        experience_benchmark = None

        if selected_experience:
            experience_mask = (
                role_df["Experience_Level"]
                .astype(str)
                .str.strip()
                .str.casefold()
                == selected_experience.casefold()
            )
            experience_df = role_df[experience_mask]

            if not experience_df.empty:
                experience_benchmark = {
                    "level": selected_experience,
                    "average": round(
                        float(experience_df["Salary_LPA"].mean()), 2
                    ),
                    "median": round(
                        float(experience_df["Salary_LPA"].median()), 2
                    ),
                    "records": int(len(experience_df)),
                }

        # 1. Highest-paying cities.
        city_salary = (
            role_df.dropna(subset=["City"])
            .groupby("City")
            .agg(
                average_salary=("Salary_LPA", "mean"),
                job_records=("Salary_LPA", "size"),
                openings=("Openings", "sum"),
            )
            .reset_index()
        )

        city_salary = city_salary[
            city_salary["City"].astype(str).str.strip() != ""
        ]

        # Show record counts so small samples are visible to users.
        city_salary = city_salary.sort_values(
            ["average_salary", "job_records"],
            ascending=[False, False],
        ).head(8)

        city_labels = city_salary["City"].astype(str).tolist()
        city_salary_values = [
            round(float(v), 2)
            for v in city_salary["average_salary"].tolist()
        ]
        city_record_counts = [
            int(v) for v in city_salary["job_records"].tolist()
        ]

        # 2. Salary by experience.
        experience_data = (
            role_df.dropna(subset=["Experience_Level"])
            .groupby("Experience_Level")
            .agg(
                average_salary=("Salary_LPA", "mean"),
                median_salary=("Salary_LPA", "median"),
                job_records=("Salary_LPA", "size"),
            )
            .reset_index()
            .sort_values("average_salary")
        )

        experience_labels = (
            experience_data["Experience_Level"].astype(str).tolist()
        )
        experience_salary_values = [
            round(float(v), 2)
            for v in experience_data["average_salary"].tolist()
        ]
        experience_median_values = [
            round(float(v), 2)
            for v in experience_data["median_salary"].tolist()
        ]
        experience_record_counts = [
            int(v) for v in experience_data["job_records"].tolist()
        ]

        # 3. Hiring demand by city.
        hiring_data = (
            role_df.dropna(subset=["City"])
            .groupby("City")
            .agg(
                openings=("Openings", "sum"),
                applicants=("Applicants", "sum"),
                job_records=("Job_Title", "size"),
            )
            .reset_index()
        )

        hiring_data = hiring_data[
            hiring_data["City"].astype(str).str.strip() != ""
        ].sort_values("openings", ascending=False).head(8)

        hiring_labels = hiring_data["City"].astype(str).tolist()
        hiring_openings = [
            int(v) for v in hiring_data["openings"].tolist()
        ]
        hiring_applicants = [
            int(v) for v in hiring_data["applicants"].tolist()
        ]
        hiring_records = [
            int(v) for v in hiring_data["job_records"].tolist()
        ]
        applicants_per_opening = [
            round(float(a / o), 2) if o > 0 else None
            for a, o in zip(hiring_applicants, hiring_openings)
        ]

        total_openings = int(role_df["Openings"].sum())
        total_applicants = int(role_df["Applicants"].sum())

        # 4. Frequently listed skills and associated salary.
        skill_df = role_df.dropna(subset=["Skills_Required"]).copy()
        skill_df["Skill"] = (
            skill_df["Skills_Required"]
            .astype(str)
            .str.split(r"[,;|]")
        )
        skill_df = skill_df.explode("Skill")
        skill_df["Skill"] = skill_df["Skill"].astype(str).str.strip()
        skill_df = skill_df[
            (skill_df["Skill"] != "")
            & (skill_df["Skill"].str.casefold() != "nan")
        ]

        # Count a skill at most once per job record using unique record ID.
        skill_df = skill_df.drop_duplicates(
            subset=["_record_id", "Skill"]
        )

        skills_data = (
            skill_df.groupby("Skill")
            .agg(
                job_records=("Salary_LPA", "size"),
                average_salary=("Salary_LPA", "mean"),
            )
            .reset_index()
            .sort_values("job_records", ascending=False)
            .head(10)
        )

        skill_labels = skills_data["Skill"].astype(str).tolist()
        skill_record_counts = [
            int(v) for v in skills_data["job_records"].tolist()
        ]
        skill_salary_values = [
            round(float(v), 2)
            for v in skills_data["average_salary"].tolist()
        ]

        # 5. Work mode and job type comparisons.
        work_mode_data = (
            role_df.dropna(subset=["Work_Mode"])
            .groupby("Work_Mode")
            .agg(
                average_salary=("Salary_LPA", "mean"),
                job_records=("Salary_LPA", "size"),
            )
            .reset_index()
            .sort_values("average_salary", ascending=False)
        )

        work_mode_labels = (
            work_mode_data["Work_Mode"].astype(str).tolist()
        )
        work_mode_salary_values = [
            round(float(v), 2)
            for v in work_mode_data["average_salary"].tolist()
        ]
        work_mode_record_counts = [
            int(v) for v in work_mode_data["job_records"].tolist()
        ]

        job_type_data = (
            role_df.dropna(subset=["Job_Type"])
            .groupby("Job_Type")
            .agg(
                average_salary=("Salary_LPA", "mean"),
                job_records=("Salary_LPA", "size"),
            )
            .reset_index()
            .sort_values("average_salary", ascending=False)
        )

        job_type_labels = job_type_data["Job_Type"].astype(str).tolist()
        job_type_salary_values = [
            round(float(v), 2)
            for v in job_type_data["average_salary"].tolist()
        ]
        job_type_record_counts = [
            int(v) for v in job_type_data["job_records"].tolist()
        ]

        context = {
            "job_title": job_title,
            "selected_experience": selected_experience,
            "predicted_salary": predicted_salary,
            "salary_difference": (
                round(salary_difference, 2)
                if salary_difference is not None else None
            ),
            "salary_difference_percent": (
                round(salary_difference_percent, 2)
                if salary_difference_percent is not None else None
            ),
            "average_salary": round(average_salary, 2),
            "median_salary": round(median_salary, 2),
            "minimum_salary": round(minimum_salary, 2),
            "maximum_salary": round(maximum_salary, 2),
            "total_records": total_records,
            "total_openings": total_openings,
            "total_applicants": total_applicants,
            "experience_benchmark": experience_benchmark,

            "city_labels": city_labels,
            "city_salary_values": city_salary_values,
            "city_record_counts": city_record_counts,

            "experience_labels": experience_labels,
            "experience_salary_values": experience_salary_values,
            "experience_median_values": experience_median_values,
            "experience_record_counts": experience_record_counts,

            "hiring_labels": hiring_labels,
            "hiring_openings": hiring_openings,
            "hiring_applicants": hiring_applicants,
            "hiring_records": hiring_records,
            "applicants_per_opening": applicants_per_opening,

            "skill_labels": skill_labels,
            "skill_record_counts": skill_record_counts,
            "skill_salary_values": skill_salary_values,

            "work_mode_labels": work_mode_labels,
            "work_mode_salary_values": work_mode_salary_values,
            "work_mode_record_counts": work_mode_record_counts,

            "job_type_labels": job_type_labels,
            "job_type_salary_values": job_type_salary_values,
            "job_type_record_counts": job_type_record_counts,
        }
        return render(request, "analytics.html", context)
    except (OSError, ValueError):
        logger.exception("Unable to generate job market analytics")
        return render(
            request,
            "analytics.html",
            {
                "job_title": job_title,
                "error_message": (
                    "Unable to load job market analytics. "
                    "Please try again later."
                ),
            },
        )



@staff_member_required
def admin_dashboard(request):
    # -------------------------------
    # Dashboard statistics
    # -------------------------------
    total_users = User.objects.filter(is_staff=False).count()
    total_predictions = PredictionHistory.objects.count()

    average_salary = PredictionHistory.objects.aggregate(
        average=Avg("predicted_salary")
    )["average"]

    # -------------------------------
    # Search and date filters
    # -------------------------------
    search_query = request.GET.get("q", "").strip()
    date_from = request.GET.get("date_from", "").strip()
    date_to = request.GET.get("date_to", "").strip()

    filtered_predictions = PredictionHistory.objects.select_related("user")

    if search_query:
        filtered_predictions = filtered_predictions.filter(
            Q(user__username__icontains=search_query)
            | Q(job_title__icontains=search_query)
        )

    # Apply only valid ISO dates
    from datetime import date

    try:
        if date_from:
            date.fromisoformat(date_from)
            filtered_predictions = filtered_predictions.filter(
                created_at__date__gte=date_from
            )
    except ValueError:
        date_from = ""

    try:
        if date_to:
            date.fromisoformat(date_to)
            filtered_predictions = filtered_predictions.filter(
                created_at__date__lte=date_to
            )
    except ValueError:
        date_to = ""

    recent_predictions = filtered_predictions.order_by(
        "-created_at"
    )[:10]

    # -------------------------------
    # Prediction activity chart
    # Last 7 days, including today
    # -------------------------------
    today = timezone.localdate()
    start_date = today - timedelta(days=6)

    activity_rows = (
        PredictionHistory.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=today,
        )
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(total=Count("id"))
        .order_by("day")
    )

    activity_map = {
        row["day"].isoformat(): row["total"]
        for row in activity_rows
    }

    activity_labels = []
    activity_data = []

    for offset in range(7):
        day = start_date + timedelta(days=offset)
        activity_labels.append(day.strftime("%d %b"))
        activity_data.append(activity_map.get(day.isoformat(), 0))

    # -------------------------------
    # Salary distribution chart
    # -------------------------------
    salary_values = PredictionHistory.objects.values_list(
        "predicted_salary", flat=True
    )

    salary_bands = [
        ("Below ₹5 LPA", 0, 5),
        ("₹5–10 LPA", 5, 10),
        ("₹10–20 LPA", 10, 20),
        ("₹20–40 LPA", 20, 40),
        ("₹40 LPA and above", 40, None),
    ]

    salary_labels = []
    salary_data = []

    for label, lower, upper in salary_bands:
        salary_labels.append(label)

        if upper is None:
            count = sum(
                1 for salary in salary_values
                if salary is not None and salary >= lower
            )
        else:
            count = sum(
                1 for salary in salary_values
                if salary is not None and lower <= salary < upper
            )

        salary_data.append(count)

    context = {
        "page_title": "Admin Dashboard",
        "total_users": total_users,
        "total_predictions": total_predictions,
        "average_salary": average_salary,
        "recent_predictions": recent_predictions,
        "search_query": search_query,
        "date_from": date_from,
        "date_to": date_to,
        "activity_labels_json": json.dumps(activity_labels),
        "activity_data_json": json.dumps(activity_data),
        "salary_labels_json": json.dumps(salary_labels),
        "salary_data_json": json.dumps(salary_data),
    }

    return render(request, "admin_dashboard.html", context)




@staff_member_required
def all_predictions(request):
    search_query = request.GET.get("q", "").strip()
    date_from = request.GET.get("date_from", "").strip()
    date_to = request.GET.get("date_to", "").strip()

    predictions = PredictionHistory.objects.select_related("user")

    # Search by username, job title, company, or city
    if search_query:
        predictions = predictions.filter(
            Q(user__username__icontains=search_query)
            | Q(job_title__icontains=search_query)
            | Q(company__icontains=search_query)
            | Q(city__icontains=search_query)
        )

    # Validate and apply date filters
    from datetime import date

    try:
        if date_from:
            date.fromisoformat(date_from)
            predictions = predictions.filter(created_at__date__gte=date_from)
    except ValueError:
        date_from = ""

    try:
        if date_to:
            date.fromisoformat(date_to)
            predictions = predictions.filter(created_at__date__lte=date_to)
    except ValueError:
        date_to = ""

    predictions = predictions.order_by("-created_at")

    context = {
        "page_title": "All Predictions",
        "predictions": predictions,
        "search_query": search_query,
        "date_from": date_from,
        "date_to": date_to,
        "total_predictions": predictions.count(),
    }

    return render(request, "all_predictions.html", context)



@staff_member_required
def manage_users(request):
    users = User.objects.all().order_by("-date_joined")

    return render(
        request,
        "manage_users.html",
        {
            "page_title": "User Management",
            "users": users,
        },
    )


@staff_member_required
@require_POST
def toggle_user_status(request, user_id):
    target_user = get_object_or_404(User, pk=user_id)

    # Protect staff and superuser accounts
    if target_user.is_staff or target_user.is_superuser:
        messages.error(
            request,
            "Staff and superuser accounts cannot be changed here.",
        )
        return redirect("manage_users")

    target_user.is_active = not target_user.is_active
    target_user.save(update_fields=["is_active"])

    if target_user.is_active:
        messages.success(
            request,
            f"User '{target_user.username}' has been activated.",
        )
    else:
        messages.success(
            request,
            f"User '{target_user.username}' has been deactivated.",
        )

    return redirect("manage_users")


def history(request):
    if request.user.is_authenticated:
        predictions = PredictionHistory.objects.filter(
            user=request.user
        )

        return render(
            request,
            "history.html",
            {
                "page_title": "Prediction History",
                "predictions": predictions,
                "is_guest": False,
            },
        )

    return render(
        request,
        "history.html",
        {
            "page_title": "Prediction History",
            "predictions": [],
            "is_guest": True,
        },
    )


def register_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = UserCreationForm(request.POST)

        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(
                request,
                "Your SalaryScope account has been created successfully!",
            )
            return redirect("home")
    else:
        form = UserCreationForm()

    return render(
        request,
        "register.html",
        {"form": form, "page_title": "Create Account"},
    )


from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:
            login(request, user)

            if user.is_staff:
                return redirect("admin_dashboard")

            return redirect("home")

        return render(request, "login.html", {
            "error": "Invalid username or password"
        })

    return render(request, "login.html")


@login_required
def logout_view(request):
    if request.method == "POST":
        logout(request)
        messages.success(request, "You have logged out successfully.")
        return redirect("home")

    return redirect("home")
