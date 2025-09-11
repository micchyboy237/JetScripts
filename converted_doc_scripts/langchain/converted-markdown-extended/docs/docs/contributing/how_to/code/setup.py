from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Setup

This guide walks through how to run the repository locally and check in your first code.
For a [development container](https://containers.dev/), see the [.devcontainer folder](https://github.com/langchain-ai/langchain/tree/master/.devcontainer).

## Dependency management: `uv` and other env/dependency managers

This project utilizes [uv](https://docs.astral.sh/uv/) v0.5+ as a dependency manager.

Install `uv`: **[documentation on how to install it](https://docs.astral.sh/uv/getting-started/installation/)**.

### Windows Users

If you're on Windows and don't have `make` installed, you can install it via:
- **Option 1**: Install via [Chocolatey](https://chocolatey.org/): `choco install make`
- **Option 2**: Install via [Scoop](https://scoop.sh/): `scoop install make`
- **Option 3**: Use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/)
- **Option 4**: Use the direct `uv` commands shown in the sections below

## Different packages

This repository contains multiple packages:
- `langchain-core`: Base interfaces for key abstractions as well as logic for combining them in chains (LangChain Expression Language).
- `langchain`: Chains, agents, and retrieval logic that makes up the cognitive architecture of your applications.
- Partner integrations: Partner packages in `libs/partners` that are independently version controlled.

:::note

Some LangChain packages live outside the monorepo, see for example
[langchain-community](https://github.com/langchain-ai/langchain-community) for various
third-party integrations and
[langchain-experimental](https://github.com/langchain-ai/langchain-experimental) for
abstractions that are experimental (either in the sense that the techniques are novel
and still being tested, or they require giving the LLM more access than would be
possible in most production systems).

:::

Each of these has its own development environment. Docs are run from the top-level makefile, but development
is split across separate test & release flows.

For this quickstart, start with `langchain`:
"""
logger.info("# Setup")

cd libs/langchain

"""
## Local development dependencies

Install development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):
"""
logger.info("## Local development dependencies")

uv sync

"""
Then verify dependency installation:
"""
logger.info("Then verify dependency installation:")

make test

uv run --group test pytest -n auto --disable-socket --allow-unix-socket tests/unit_tests

"""
## Testing

**Note:** In `langchain`, `langchain-community`, and `langchain-experimental`, some test dependencies are optional. See the following section about optional dependencies.

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.

To run unit tests:
"""
logger.info("## Testing")

make test

uv run --group test pytest -n auto --disable-socket --allow-unix-socket tests/unit_tests

"""
There are also [integration tests and code-coverage](../testing.mdx) available.

### Developing langchain_core

If you are only developing `langchain_core`, you can simply install the dependencies for the project and run tests:
"""
logger.info("### Developing langchain_core")

cd libs/core

make test

uv run --group test pytest -n auto --disable-socket --allow-unix-socket tests/unit_tests

"""
## Formatting and linting

Run these locally before submitting a PR; the CI system will check also.

### Code formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for docs, cookbook and templates:
"""
logger.info("## Formatting and linting")

make format

uv run --all-groups ruff format .
uv run --all-groups ruff check --fix .

"""
To run formatting for a library, run the same command from the relevant library directory:
"""
logger.info("To run formatting for a library, run the same command from the relevant library directory:")

cd libs/{LIBRARY}

make format

uv run --all-groups ruff format .
uv run --all-groups ruff check --fix .

"""
Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:
"""
logger.info("Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:")

make format_diff

git diff --relative=libs/langchain --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs uv run --all-groups ruff format
git diff --relative=libs/langchain --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs uv run --all-groups ruff check --fix

"""
This is especially useful when you have made changes to a subset of the project and want to ensure your changes are properly formatted without affecting the rest of the codebase.

#### Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run linting for docs, cookbook and templates:
"""
logger.info("#### Linting")

make lint

uv run --all-groups ruff check .
uv run --all-groups ruff format . --diff
uv run --all-groups mypy . --cache-dir .mypy_cache

"""
To run linting for a library, run the same command from the relevant library directory:
"""
logger.info("To run linting for a library, run the same command from the relevant library directory:")

cd libs/{LIBRARY}

make lint

uv run --all-groups ruff check .
uv run --all-groups ruff format . --diff
uv run --all-groups mypy . --cache-dir .mypy_cache

"""
In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:
"""
logger.info("In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:")

make lint_diff

git diff --relative=libs/langchain --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs uv run --all-groups ruff check
git diff --relative=libs/langchain --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs uv run --all-groups ruff format --diff
git diff --relative=libs/langchain --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs uv run --all-groups mypy --cache-dir .mypy_cache

"""
This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

### Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so it could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:
"""
logger.info("### Spellcheck")

make spell_check

uv run --all-groups codespell --toml pyproject.toml

"""
To fix spelling in place:
"""
logger.info("To fix spelling in place:")

make spell_fix

uv run --all-groups codespell --toml pyproject.toml -w

"""
If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.
"""
logger.info("If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.")

[tool.codespell]
...
ignore-words-list = 'momento,collison,ned,foor,reworkd,parth,whats,aapply,mysogyny,unsecure'

"""
### Pre-commit

We use [pre-commit](https://pre-commit.com/) to ensure commits are formatted/linted.

#### Installing Pre-commit

First, install pre-commit:
"""
logger.info("### Pre-commit")

uv tool install pre-commit

brew install pre-commit

pip install pre-commit

"""
Then install the git hook scripts:
"""
logger.info("Then install the git hook scripts:")

pre-commit install

"""
#### How Pre-commit Works

Once installed, pre-commit will automatically run on every `git commit`. Hooks are specified in `.pre-commit-config.yaml` and will:

- Format code using `ruff` for the specific library/package you're modifying
- Only run on files that have changed
- Prevent commits if formatting fails

#### Skipping Pre-commit

In exceptional cases, you can skip pre-commit hooks with:
"""
logger.info("#### How Pre-commit Works")

git commit --no-verify

"""
However, this is discouraged as the CI system will still enforce the same formatting rules.

## Working with optional dependencies

`langchain`, `langchain-community`, and `langchain-experimental` rely on optional dependencies to keep these packages lightweight.

`langchain-core` and partner packages **do not use** optional dependencies in this way.

You'll notice that `pyproject.toml` and `uv.lock` are **not** touched when you add optional dependencies below.

If you're adding a new dependency to Langchain, assume that it will be an optional dependency, and
that most users won't have it installed.

Users who do not have the dependency installed should be able to **import** your code without
any side effects (no warnings, no errors, no exceptions).

To introduce the dependency to a library, please do the following:

1. Open extended_testing_deps.txt and add the dependency
2. Add a unit test that the very least attempts to import the new code. Ideally, the unit
test makes use of lightweight fixtures to test the logic of the code.
3. Please use the `@pytest.mark.requires(package_name)` decorator for any unit tests that require the dependency.

## Adding a Jupyter Notebook

If you are adding a Jupyter Notebook example, you'll want to run with `test` dependencies:
"""
logger.info("## Working with optional dependencies")

uv run --group test jupyter notebook

"""
When you run `uv sync`, the `langchain` package is installed as editable in the virtualenv, so your new logic can be imported into the notebook.
"""
logger.info("When you run `uv sync`, the `langchain` package is installed as editable in the virtualenv, so your new logic can be imported into the notebook.")

logger.info("\n\n[DONE]", bright=True)