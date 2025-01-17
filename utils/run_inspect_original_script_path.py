from jet.logger import logger
from jet.utils.inspect_utils import inspect_original_script_path


if __name__ == "__main__":
    # Call from another module
    logger.debug("Runner's script path:")
    logger.success(inspect_original_script_path())
