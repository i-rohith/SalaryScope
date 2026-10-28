
from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "job_title",
        "company",
        "city",
        "predicted_salary",
        "created_at",
    )

    list_filter = (
        "city",
        "experience_level",
        "industry",
        "created_at",
    )

    search_fields = (
        "user__username",
        "job_title",
        "company",
        "city",
    )

    readonly_fields = ("created_at",)

    ordering = ("-created_at",)
