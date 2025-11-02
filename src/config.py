from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
RAW_TRAIN = DATA_DIR / 'raw' / 'train.csv'     # må finnes
TARGET_CANDIDATES = ['price', 'selling_price', 'target']  # sjekkes i rekkefølge
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_PATH = MODEL_DIR / 'model.pkl'
SCHEMA_PATH = MODEL_DIR / 'schema.json'
METRICS_PATH = MODEL_DIR / 'metrics.json'
RANDOM_STATE = 42
TEST_SIZE = 0.2
