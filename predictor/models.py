
from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="salary_predictions"
    )

    job_title = models.CharField(max_length=200)
    company = models.CharField(max_length=200, default="Unknown")
    company_type = models.CharField(max_length=100)
    industry = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    location_tier = models.CharField(max_length=50)
    experience_level = models.CharField(max_length=50)
    job_type = models.CharField(max_length=50)
    work_mode = models.CharField(max_length=50)
    skills_required = models.TextField()
    education_required = models.CharField(max_length=100)

    predicted_salary = models.DecimalField(
        max_digits=10,
        decimal_places=2
    )
    lower_salary = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )
    upper_salary = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )
    interval_level = models.PositiveSmallIntegerField(
        default=90
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.job_title} - {self.user.username}"
