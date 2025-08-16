

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
myst:
  html_meta:
    "description lang=en": |
      User Guide for AutoGen Studio - A low code tool for building and debugging multi-agent systems
---

# Installation

There are two ways to install AutoGen Studio - from PyPi or from source. We **recommend installing from PyPi** unless you plan to modify the source code.

## Create a Virtual Environment (Recommended)

We recommend using a virtual environment as this will ensure that the dependencies for AutoGen Studio are isolated from the rest of your system.

Create and activate:

Linux/Mac:
"""
logger.info("# Installation")

python3 -m venv .venv
source .venv/bin/activate

"""
Windows command-line:
"""
logger.info("Windows command-line:")

python3 -m venv .venv
.venv\Scripts\activate.bat

"""
To deactivate later, run:
"""
logger.info("To deactivate later, run:")

deactivate

"""


[Install Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) if you have not already.


Create and activate:
"""
logger.info("Create and activate:")

conda create -n autogen python=3.10
conda activate autogen

"""
To deactivate later, run:
"""
logger.info("To deactivate later, run:")

conda deactivate

"""


## Install from PyPi (Recommended)

You can install AutoGen Studio using pip, the Python package manager.
"""
logger.info("## Install from PyPi (Recommended)")

pip install -U autogenstudio

"""
## Install from source

_Note: This approach requires some familiarity with building interfaces in React._

You have two options for installing from source: manually or using a dev container.

### A) Install from source manually

1. Ensure you have Python 3.10+ and Node.js (version above 14.15.0) installed.
2. Clone the AutoGen Studio repository.
3. Navigate to the `python/packages/autogen-studio` and install its Python dependencies using `pip install -e .`
4. Navigate to the `python/packages/autogen-studio/frontend` directory, install the dependencies, and build the UI:
"""
logger.info("## Install from source")

npm install -g gatsby-cli
npm install --global yarn
cd frontend
yarn install
yarn build
gatsby clean && rmdir /s /q ..\\autogenstudio\\web\\ui 2>nul & (set \"PREFIX_PATH_VALUE=\" || ver>nul) && gatsby build --prefix-paths && xcopy /E /I /Y public ..\\autogenstudio\\web\\ui

"""
### B) Install from source using a dev container

1. Follow the [Dev Containers tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) to install VS Code, Docker and relevant extensions.
2. Clone the AutoGen Studio repository.
3. Open `python/packages/autogen-studio/`in VS Code. Click the blue button in bottom the corner or press F1 and select _"Dev Containers: Reopen in Container"_.
4. Build the UI:
"""
logger.info("### B) Install from source using a dev container")

cd frontend
yarn build

"""
## Running the Application

Once installed, run the web UI by entering the following in your terminal:
"""
logger.info("## Running the Application")

autogenstudio ui --port 8081

"""
This command will start the application on the specified port. Open your web browser and go to <http://localhost:8081/> to use AutoGen Studio.

AutoGen Studio also takes several parameters to customize the application:

- `--host <host>` argument to specify the host address. By default, it is set to `localhost`.
- `--appdir <appdir>` argument to specify the directory where the app files (e.g., database and generated user files) are stored. By default, it is set to the `.autogenstudio` directory in the user's home directory.
- `--port <port>` argument to specify the port number. By default, it is set to `8080`.
- `--reload` argument to enable auto-reloading of the server when changes are made to the code. By default, it is set to `False`.
- `--database-uri` argument to specify the database URI. Example values include `sqlite:///database.sqlite` for SQLite and `postgresql+psycopg://user:password@localhost/dbname` for PostgreSQL. If this is not specified, the database URL defaults to a `database.sqlite` file in the `--appdir` directory.
- `--upgrade-database` argument to upgrade the database schema to the latest version. By default, it is set to `False`.

Now that you have AutoGen Studio installed and running, you are ready to explore its capabilities, including defining and modifying agent workflows, interacting with agents and sessions, and expanding agent skills.
"""
logger.info("This command will start the application on the specified port. Open your web browser and go to <http://localhost:8081/> to use AutoGen Studio.")

logger.info("\n\n[DONE]", bright=True)