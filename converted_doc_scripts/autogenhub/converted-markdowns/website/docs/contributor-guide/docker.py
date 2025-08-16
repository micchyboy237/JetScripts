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
# Docker for Development

For developers contributing to the AutoGen project, we offer a specialized Docker environment. This setup is designed to streamline the development process, ensuring that all contributors work within a consistent and well-equipped environment.

## Autogen Developer Image (autogenhub_dev_img)

- **Purpose**: The `autogenhub_dev_img` is tailored for contributors to the AutoGen project. It includes a suite of tools and configurations that aid in the development and testing of new features or fixes.
- **Usage**: This image is recommended for developers who intend to contribute code or documentation to AutoGen.
- **Forking the Project**: It's advisable to fork the AutoGen GitHub project to your own repository. This allows you to make changes in a separate environment without affecting the main project.
- **Updating Dockerfile**: Modify your copy of `Dockerfile` in the `dev` folder as needed for your development work.
- **Submitting Pull Requests**: Once your changes are ready, submit a pull request from your branch to the upstream AutoGen GitHub project for review and integration. For more details on contributing, see the [AutoGen Contributing](https://autogenhub.github.io/autogen/docs/Contribute) page.

## Building the Developer Docker Image

- To build the developer Docker image (`autogenhub_dev_img`), use the following commands:

  ```bash
  docker build -f .devcontainer/dev/Dockerfile -t autogenhub_dev_img https://github.com/autogenhub/autogen.git#main
  ```

- For building the developer image built from a specific Dockerfile in a branch other than main/master

  ```bash
  # clone the branch you want to work out of
  git clone --branch {branch-name} https://github.com/autogenhub/autogen.git

  # cd to your new directory
  cd autogen

  # build your Docker image
  docker build -f .devcontainer/dev/Dockerfile -t autogen_dev-srv_img .
  ```

## Using the Developer Docker Image

Once you have built the `autogenhub_dev_img`, you can run it using the standard Docker commands. This will place you inside the containerized development environment where you can run tests, develop code, and ensure everything is functioning as expected before submitting your contributions.
"""
logger.info("# Docker for Development")

docker run -it -p 8081:3000 -v `pwd`/autogen-newcode:newstuff/ autogenhub_dev_img bash

"""
- Note that the `pwd` is shorthand for present working directory. Thus, any path after the pwd is relative to that. If you want a more verbose method you could remove the "`pwd`/autogen-newcode" and replace it with the full path to your directory
"""

docker run -it -p 8081:3000 -v /home/AutoGenDeveloper/autogen-newcode:newstuff/ autogenhub_dev_img bash

"""
## Develop in Remote Container

If you use vscode, you can open the autogen folder in a [Container](https://code.visualstudio.com/docs/remote/containers).
We have provided the configuration in [devcontainer](https://github.com/autogenhub/autogen/blob/main/.devcontainer). They can be used in GitHub codespace too. Developing AutoGen in dev containers is recommended.
"""
logger.info("## Develop in Remote Container")

logger.info("\n\n[DONE]", bright=True)