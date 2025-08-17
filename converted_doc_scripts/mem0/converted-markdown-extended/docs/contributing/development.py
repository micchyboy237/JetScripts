from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Development
icon: "code"
---

# Development Contributions

We strive to make contributions **easy, collaborative, and enjoyable**. Follow the steps below to ensure a smooth contribution process.

## Submitting Your Contribution through PR

To contribute, follow these steps:

1. **Fork & Clone** the repository: [Mem0 on GitHub](https://github.com/mem0ai/mem0)
2. **Create a Feature Branch**: Use a dedicated branch for your changes, e.g., `feature/my-new-feature`
3. **Implement Changes**: If adding a feature or fixing a bug, ensure to:
   - Write necessary **tests**
   - Add **documentation, docstrings, and runnable examples**
4. **Code Quality Checks**:
   - Run **linting** to catch style issues
   - Ensure **all tests pass**
5. **Submit a Pull Request** 🚀

For detailed guidance on pull requests, refer to [GitHub's documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

---

## 📦 Dependency Management

We use `hatch` as our package manager. Install it by following the [official instructions](https://hatch.pypa.io/latest/install/).

⚠️ **Do NOT use `pip` or `conda` for dependency management.** Instead, follow these steps in order:
"""
logger.info("# Development Contributions")

make install

hatch shell (for default env)
hatch -e dev_py_3_11 shell (for dev_py_3_11) (differences are mentioned in pyproject.toml)

make install_all

"""
---

## 🛠️ Development Standards

### ✅ Pre-commit Hooks

Ensure `pre-commit` is installed before contributing:
"""
logger.info("## 🛠️ Development Standards")

pre-commit install

"""
### 🔍 Linting with `ruff`

Run the linter and fix any reported issues before submitting your PR:
"""
logger.info("### 🔍 Linting with `ruff`")

make lint

"""
### 🎨 Code Formatting

To maintain a consistent code style, format your code:
"""
logger.info("### 🎨 Code Formatting")

make format

"""
### 🧪 Testing with `pytest`

Run tests to verify functionality before submitting your PR:
"""
logger.info("### 🧪 Testing with `pytest`")

make test

"""
💡 **Note:** Some dependencies have been removed from the main dependencies to reduce package size. Run `make install_all` to install necessary dependencies before running tests.

---

## 🚀 Release Process

Currently, releases are handled manually. We aim for frequent releases, typically when new features or bug fixes are introduced.

---

Thank you for contributing to Mem0! 🎉
"""
logger.info("## 🚀 Release Process")

logger.info("\n\n[DONE]", bright=True)