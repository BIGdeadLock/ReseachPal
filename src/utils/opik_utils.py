import os

import opik
from loguru import logger
from opik.configurator.configure import OpikConfigurator

from src.utils.config import config as settings

def configure_opik() -> None:
    if settings.opik.api_key and settings.opik.project:
        try:
            client = OpikConfigurator(api_key=settings.opik.api_key)
            default_workspace = client._get_default_workspace()
        except Exception:
            logger.warning("Default workspace not found. Setting workspace to None and enabling interactive mode.")
            default_workspace = None

        os.environ["OPIK_PROJECT_NAME"] = settings.opik.project

        opik.configure(api_key=settings.opik.api_key, workspace=default_workspace, use_local=False, force=True)
        logger.info("Opik configured successfully.")
    else:
        logger.warning(
            "COMET_API_KEY and COMET_PROJECT are not set. Set them to enable prompt monitoring with Opik (powered by Comet ML)."
        )