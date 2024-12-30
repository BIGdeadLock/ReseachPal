import os, sys

WINDOWS_OS_STR = "nt"
IS_WINDOWS_OS = (os.name == WINDOWS_OS_STR)

PROJECT_DIR = os.environ['PROJECT_DIR'] if 'PROJECT_DIR' in os.environ else os.path.dirname(
    sys.modules['__main__'].__file__)

# -----------------
# FILE PATHS
# -----------------
OUTPUT_DIR = 'output'
OUTPUT_DIR_PATH = os.path.join(PROJECT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

REPORT_FILE = "report.md"
REPORT_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, REPORT_FILE)

STORE_DIR = "store"
STORE_PATH = os.path.join(PROJECT_DIR, STORE_DIR)
os.makedirs(STORE_PATH, exist_ok=True)

MODELS_DIR = "models"
MODELS_PATH = os.path.join(STORE_PATH, MODELS_DIR)
os.makedirs(MODELS_PATH, exist_ok=True)

ENV_FILE = '.env'
ENV_FILE_PATH = os.path.join(PROJECT_DIR, ENV_FILE)

# -----------------
# RETRIEVERS
# -----------------
TOP_K_DEFAULT = 20
KEEP_TOP_K_DEFAULT = 5
RELEVANT_SCORE_DEFAULT = 0


# -----------------
# AGENTS
# -----------------
OPENAI_GPT4O_DEPLOYMENT_ID = "gpt-4o-deployment"
OPENAI_GPT4O_MINI_DEPLOYMENT_ID = "gpt-4o-mini-deployment"
OPENAI_API_GPTO_VERSION = '2024-06-01'