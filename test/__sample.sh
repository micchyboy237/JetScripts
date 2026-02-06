# Complete the shell script that will create the file structure with full code given this discussion
# cd <path_to_base_dir>
# source /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.sh

#!/bin/bash

# This script builds the tests/ directory and populates all main code files according to the BDD-style, pytest-based unit test suite above.
# Run: bash __sample.sh
# (Run from the project root, or adjust path as needed.)

set -e

mkdir -p tests

# --- tests/test_utils.py ---
cat > tests/test_utils.py <<'EOF'
import os
from codependpy.utils import is_third_party, resolve_relative_import, attr_chain
import ast


def test_is_third_party():
    # Given paths that should be considered third-party
    assert is_third_party("/usr/lib/python3.11/site-packages/requests/models.py") is True
    assert is_third_party("/home/user/.venv/lib/python3.10/dist-packages/numpy/__init__.py") is True

    # When paths are project-local
    assert is_third_party("/home/user/my_project/app/models.py") is False
    assert is_third_party("src/utils.py") is False


def test_resolve_relative_import():
    # Given a base file and different relative import levels
    base = "/project/myapp/views.py"

    # level 1, no module → should go up one and use empty module
    assert resolve_relative_import(base, 1, "") == "myapp"

    # level 1 with module
    assert resolve_relative_import(base, 1, "models") == "myapp.models"

    # level 2
    assert resolve_relative_import(base, 2, "utils.helpers") == "utils.helpers"

    # level 0 (absolute import fallback)
    assert resolve_relative_import(base, 0, "django.db.models") == "django.db.models"

    # edge: level higher than path parts
    assert resolve_relative_import(base, 5, "something") == "something"


def test_attr_chain():
    # Given simple Name
    node = ast.Name(id="User")
    assert attr_chain(node) == "User"

    # Given nested Attribute
    node = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="myapp"),
            attr="models"
        ),
        attr="User"
    )
    assert attr_chain(node) == "myapp.models.User"

    # Deeper chain
    node = ast.Attribute(
        value=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="api"),
                attr="v1"
            ),
            attr="views"
        ),
        attr="UserView"
    )
    assert attr_chain(node) == "api.v1.views.UserView"
EOF

# --- tests/test_file_scanner.py ---
cat > tests/test_file_scanner.py <<'EOF'
from codependpy.file_scanner import scan_python_files
import os
import tempfile


def test_scan_python_files_finds_py_files(tmp_path):
    # Given a temporary directory structure
    (tmp_path / "app" / "models.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "models.py").write_text("class User: pass")
    (tmp_path / "app" / "views.py").write_text("def view(): pass")
    (tmp_path / "tests" / "test_models.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_models.py").write_text("def test_something(): pass")

    # When scanning without exclude
    files = scan_python_files([str(tmp_path)], exclude="")

    # Then should find both .py files
    assert len(files) == 2
    assert any("models.py" in f for f in files)
    assert any("views.py" in f for f in files)


def test_scan_respects_exclude_regex(tmp_path):
    # Given files in excluded and non-excluded dirs
    (tmp_path / "app" / "models.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "models.py").write_text("")
    (tmp_path / "migrations" / "0001_initial.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "migrations" / "0001_initial.py").write_text("")

    # When scanning with exclude
    files = scan_python_files([str(tmp_path)], exclude="migrations")

    # Then migrations should be skipped
    assert len(files) == 1
    assert "models.py" in files[0]
    assert "0001_initial.py" not in files[0]


def test_scan_skips_site_packages(tmp_path):
    # Given a fake site-packages
    (tmp_path / "venv" / "lib" / "site-packages" / "requests" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "venv" / "lib" / "site-packages" / "requests" / "__init__.py").write_text("")

    files = scan_python_files([str(tmp_path)], exclude="")

    assert len(files) == 0
EOF

# --- tests/test_parse_args.py ---
cat > tests/test_parse_args.py <<'EOF'
from codependpy.parse_args import build_targets, TargetSpec
import os


def test_build_targets_with_symbols():
    # Given CLI-style target strings
    args = [
        "app/models.py:User,Profile",
        "app/views.py:CreateView",
        "utils/helpers.py",           # no symbols
    ]

    # When converting
    targets = build_targets(args)

    # Then
    assert len(targets) == 3
    assert targets[0] == TargetSpec(
        file_path=os.path.abspath("app/models.py"),
        symbols=["User", "Profile"]
    )
    assert targets[1] == TargetSpec(
        file_path=os.path.abspath("app/views.py"),
        symbols=["CreateView"]
    )
    assert targets[2] == TargetSpec(
        file_path=os.path.abspath("utils/helpers.py"),
        symbols=[]
    )
EOF

# --- tests/test_index_builder.py ---
cat > tests/test_index_builder.py <<'EOF'
from codependpy.index_builder import to_module_name, build_parsed_project
from codependpy.project_types import FileInfo, ParsedProject


def test_to_module_name():
    base_dirs = ["/home/user/project", "/home/user/shared"]

    # Simple case
    assert to_module_name("/home/user/project/app/models.py", base_dirs) == "app.models"

    # Deeper path
    assert to_module_name("/home/user/project/src/api/v1/views.py", base_dirs) == "src.api.v1.views"

    # Outside base → fallback to basename
    assert to_module_name("/other/lib/utils.py", base_dirs) == "utils"


def test_build_parsed_project_creates_indexes(tmp_path):
    f1 = tmp_path / "models.py"
    f1.write_text("""
class User:
    pass

class Profile:
    pass
""")

    f2 = tmp_path / "views.py"
    f2.write_text("""
from .models import User

def create_user():
    pass
""")

    files = {
        str(f1): FileInfo(path=str(f1)),
        str(f2): FileInfo(path=str(f2)),
    }

    # Normally you'd parse them — here we mock minimal definitions
    files[str(f1)].definitions = {"User": None, "Profile": None}
    files[str(f2)].definitions = {"create_user": None}

    project = build_parsed_project(files, [str(tmp_path)])

    assert project.symbol_index == {
        "User": [str(f1)],
        "Profile": [str(f1)],
        "create_user": [str(f2)],
    }

    assert project.module_index == {
        "models": str(f1),
        "views": str(f2),
    }
EOF

# --- tests/test_code_reconstructor.py ---
cat > tests/test_code_reconstructor.py <<'EOF'
from codependpy.code_reconstructor import reconstruct_code
from codependpy.project_types import FileInfo, Definition


def test_reconstruct_top_level_functions_only():
    # Given a file with two functions
    source = [
        "import os\n",
        "\n",
        "def helper():\n",
        "    pass\n",
        "\n",
        "def main():\n",
        "    pass\n",
    ]

    info = FileInfo(path="utils.py")
    info.source_lines = source
    info.definitions = {
        "helper": Definition("helper", 3, 4, None),
        "main": Definition("main", 6, 7, None),
    }

    needed = {"utils.py": {"helper"}}

    # When reconstructing
    result = reconstruct_code(needed, {"utils.py": info})

    # Then should include import + only helper
    code = result["utils.py"]
    assert "import os" in code
    assert "def helper():" in code
    assert "def main():" not in code
EOF

# --- tests/test_ast_parser.py ---
cat > tests/test_ast_parser.py <<'EOF'
import pytest
from codependpy.ast_parser import parse_file, Parser, FunctionBodyVisitor
from codependpy.project_types import FileInfo, Definition


@pytest.fixture
def simple_file(tmp_path):
    content = """
import os as operating_system
from pathlib import Path

class User:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello {self.name}")

def standalone():
    p = Path(".")
    return operating_system.getcwd()
"""
    path = tmp_path / "simple.py"
    path.write_text(content)
    return path


def test_parse_file_collects_definitions_and_imports(simple_file):
    # Given a simple Python file
    fi: FileInfo = parse_file(str(simple_file))

    # Then definitions are collected correctly
    assert "User" in fi.definitions
    assert "User.greet" in fi.definitions
    assert "standalone" in fi.definitions

    user_def = fi.definitions["User"]
    assert user_def.start_line == 4
    assert user_def.parent is None

    greet_def = fi.definitions["User.greet"]
    assert greet_def.parent == "User"

    # And imports are tracked
    assert fi.import_aliases["operating_system"] == "os"
    assert fi.import_symbols["Path"] == ("pathlib", "Path")


def test_function_body_visitor_collects_self_references(simple_file):
    # Given parsed file info
    fi = parse_file(str(simple_file))
    greet_def = fi.definitions["User.greet"]

    # When we look at collected local references in method body
    refs = greet_def.local_refs

    # Then self.name is turned into "User.name"
    assert "User.name" in refs


def test_django_get_model_reference(tmp_path):
    content = """
from django.apps import apps

def get_user_model():
    return apps.get_model("auth", "User")
"""
    path = tmp_path / "utils.py"
    path.write_text(content)

    fi = parse_file(str(path))

    # Then django_refs should contain the tuple
    assert fi.django_refs == [("auth", "User")]


def test_url_path_reference(tmp_path):
    content = """
from django.urls import path
from .views import home_view, detail_view

urlpatterns = [
    path("", home_view, name="home"),
    path("detail/<int:pk>/", detail_view, name="detail"),
]
"""
    path = tmp_path / "urls.py"
    path.write_text(content)

    fi = parse_file(str(path))

    # Then url_refs contains the view names
    assert set(fi.url_refs) == {"home_view", "detail_view"}
EOF

# --- tests/test_dependency_bfs.py ---
cat > tests/test_dependency_bfs.py <<'EOF'
from collections import defaultdict
from codependpy.dependency_bfs import gather_definitions
from codependpy.project_types import ParsedProject, FileInfo, Definition, TargetSpec


def test_gather_single_target_no_deps():
    # Given minimal project
    files = {
        "/project/models.py": FileInfo(path="/project/models.py")
    }
    files["/project/models.py"].definitions = {
        "User": Definition("User", 1, 10, None, local_refs=set())
    }

    project = ParsedProject(
        files=files,
        symbol_index={"User": ["/project/models.py"]},
        module_index={"models": "/project/models.py"},
        django_index={}
    )

    targets = (TargetSpec(file_path="/project/models.py", symbols=["User"]),)

    # When gathering dependencies
    needed = gather_definitions(project, targets, mode="all", max_depth=5)

    # Then only that definition is needed
    assert needed == {"/project/models.py": {"User"}}


def test_collects_local_references():
    # Given a function that references another symbol
    files = {
        "/project/views.py": FileInfo(path="/project/views.py"),
        "/project/models.py": FileInfo(path="/project/models.py"),
    }

    files["/project/views.py"].definitions = {
        "create_user": Definition(
            "create_user", 5, 15, None,
            local_refs={"User", "get_user_model"}
        )
    }
    files["/project/models.py"].definitions = {
        "User": Definition("User", 3, 20, None, local_refs=set()),
        "get_user_model": Definition("get_user_model", 25, 30, None, local_refs=set())
    }

    project = ParsedProject(
        files=files,
        symbol_index={
            "User": ["/project/models.py"],
            "get_user_model": ["/project/models.py"],
            "create_user": ["/project/views.py"],
        },
        module_index={
            "views": "/project/views.py",
            "models": "/project/models.py",
        },
        django_index={}
    )

    targets = (TargetSpec("/project/views.py", ["create_user"]),)

    # When we run BFS
    needed = gather_definitions(project, targets, mode="all", max_depth=5)

    # Then both referenced symbols are collected
    assert needed["/project/views.py"] == {"create_user"}
    assert needed["/project/models.py"] == {"User", "get_user_model"}


def test_respects_max_depth():
    # Given a chain: A → B → C → D
    files = {f"/project/{n}.py": FileInfo(path=f"/project/{n}.py") for n in "ABCD"}

    files["/project/A.py"].definitions = {"A": Definition("A", 1, 2, None, {"B"})}
    files["/project/B.py"].definitions = {"B": Definition("B", 1, 2, None, {"C"})}
    files["/project/C.py"].definitions = {"C": Definition("C", 1, 2, None, {"D"})}
    files["/project/D.py"].definitions = {"D": Definition("D", 1, 2, None, set())}

    project = ParsedProject(
        files=files,
        symbol_index={n: [f"/project/{n}.py"] for n in "ABCD"},
        module_index={n: f"/project/{n}.py" for n in "ABCD"},
        django_index={}
    )

    targets = (TargetSpec("/project/A.py", ["A"]),)

    # When limiting depth to 2
    needed = gather_definitions(project, targets, mode="all", max_depth=2)

    # Then we get A, B, C but not D
    assert needed["/project/A.py"] == {"A"}
    assert needed["/project/B.py"] == {"B"}
    assert needed["/project/C.py"] == {"C"}
    assert "/project/D.py" not in needed
EOF

# --- tests/test_helpers.py ---
cat > tests/test_helpers.py <<'EOF'
from pathlib import Path
from codependpy.helpers import get_code_and_dependencies
from codependpy.project_types import TargetSpec


def test_get_code_and_dependencies_defaults_search_dir(tmp_path):
    # Given a target file and no explicit search_dirs
    target_file = tmp_path / "core" / "models.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("class User: pass")

    # When calling without search_dirs
    result = get_code_and_dependencies(
        search_dirs=[],
        exclude_regex="",
        target_file=target_file,
        target_symbols=["User"],
    )

    # Then it should use the parent directory as search root
    assert result  # at least one file processed
    assert str(target_file) in result
    assert "class User:" in result[str(target_file)]


def test_multiple_targets_via_helper(tmp_path):
    # This is more integration-like, but we can test the helper signature & basic flow
    # For full coverage, we rely on lower-level tests

    (tmp_path / "a.py").write_text("def func_a(): pass")
    (tmp_path / "b.py").write_text("def func_b(): pass")

    # Minimal call
    result = get_code_and_dependencies(
        search_dirs=[str(tmp_path)],
        exclude_regex="",
        target_file=str(tmp_path / "a.py"),
        target_symbols=["func_a"],
        bfs_mode="all",
        max_depth=2,
    )

    assert str(tmp_path / "a.py") in result
EOF

# --- Optional: conftest.py ---
touch tests/conftest.py

echo "Test files structure and code fully written to ./tests/."