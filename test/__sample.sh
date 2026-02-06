# Complete the shell script that will create the file structure with full code given this discussion
# cd <path_to_base_dir>
# source /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.sh

#!/bin/bash

# setup_mock_structure.sh
# Shell script to create the mock/ directory and minimal Django-like files,
# and update _jet_full_example.py to use mock/ as the base directory.

set -e

# Set up mock directory structure
MOCK_DIR="mock"
VIEWS_DIR="$MOCK_DIR/views"
SERVICES_DIR="$MOCK_DIR/services"

mkdir -p "$VIEWS_DIR"
mkdir -p "$SERVICES_DIR"

# Create mock/__init__.py (empty)
cat > "$MOCK_DIR/__init__.py" <<EOF
# empty
EOF

# Create mock/models.py
cat > "$MOCK_DIR/models.py" <<EOF
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

    def get_display_name(self):
        return f"{self.name} <{self.email}>"

class Project(models.Model):
    title = models.CharField(max_length=200)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)

    @classmethod
    def get_active(cls):
        return cls.objects.filter(is_active=True)
EOF

# Create mock/utils.py
cat > "$MOCK_DIR/utils.py" <<EOF
def format_date(dt):
    return dt.strftime("%Y-%m-%d")

def compute_stats(numbers):
    return {
        "total": sum(numbers),
        "count": len(numbers),
        "average": sum(numbers) / len(numbers) if numbers else 0,
    }
EOF

# Create mock/views/__init__.py (empty)
cat > "$VIEWS_DIR/__init__.py" <<EOF
# empty
EOF

# Create mock/views/dashboard.py
cat > "$VIEWS_DIR/dashboard.py" <<EOF
from django.shortcuts import render
from models import User, Project
from utils import compute_stats, format_date
from services.report import generate_pdf_report

def dashboard_view(request):
    users = User.objects.all()
    projects = Project.objects.all()
    stats = compute_stats([p.id for p in projects])
    return render(request, "dashboard.html", {"stats": stats})

def get_stats(request):
    data = [10, 20, 30, 40]
    return compute_stats(data)
EOF

# Create mock/services/__init__.py (empty)
cat > "$SERVICES_DIR/__init__.py" <<EOF
# empty
EOF

# Create mock/services/report.py
cat > "$SERVICES_DIR/report.py" <<EOF
from models import Project
from utils import format_date
from views.dashboard import get_stats  # cross-module reference example

def generate_pdf_report(project_id):
    project = Project.objects.get(id=project_id)
    stats = get_stats(None)  # dummy call
    title = f"Report for {project.title}"
    date_str = format_date(project.created_at) if hasattr(project, 'created_at') else "N/A"
    return f"{title} - {date_str}"
EOF

# Optionally apply the code diff to Utils/codependpy/_jet_full_example.py
EXAMPLE_PATH="Utils/codependpy/_jet_full_example.py"
if [ -f "$EXAMPLE_PATH" ]; then
    # Use sed to edit parse_project and targets lines
    sed -i.bak \
        -e 's|parse_project("src/")|parse_project("mock/")|' \
        -e 's|TargetSpec("src/views/dashboard.py", \["dashboard_view", "get_stats"\])|TargetSpec("mock/views/dashboard.py", ["dashboard_view", "get_stats"])|' \
        -e 's|TargetSpec("src/services/report.py", \["generate_pdf_report"\])|TargetSpec("mock/services/report.py", ["generate_pdf_report"])|' \
        "$EXAMPLE_PATH"
    echo "Patched $EXAMPLE_PATH to use mock/ as the base directory."
fi

echo "Mock project structure created. Done."
