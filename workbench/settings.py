"""settings.py -- the folders, the groupings and the fixed settings that every step reads.

Folders, all beside the scripts:
  responses/   the cohort and its answers to the 151 items (step 1)
  work/        person-level files written by the steps; they never leave the Workbench
  downloads/   the aggregate files of each step and the zip to download (downloads/<name>_<release>.zip)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CDR = os.environ.get("WORKSPACE_CDR", "")              # set by the Workbench: <project>.<dataset>
RELEASE = CDR.split(".")[-1] or "cdr"                   # the dataset, R2025Q4R6, used in the download names

ITEMS_CSV = os.path.join(HERE, "items_151.csv")         # the 151 items: order, scale, keying, instrument, construct
CODING_CSV = os.path.join(HERE, "encoding_map_all151.csv")   # every answer of every item and its numeric value
SOURCES_CSV = os.path.join(HERE, "standalone_sources.csv")   # the source study of each stand-alone question
BASIS = os.path.join(HERE, "basis")                     # the semantic groups of the 6 encoders (built from the item text)
BASIS_TFIDF = os.path.join(HERE, "basis_tfidf")         # the 2 word-count (TF-IDF) groupings
EMB_PREFIX = "admin151_items"                           # basis*/item_embeddings/<prefix>_<encoder>.npz

RESPONSES = os.path.join(HERE, "responses")
WORK = os.path.join(HERE, "work")
DOWNLOADS = os.path.join(HERE, "downloads")
COHORT_CSV = os.path.join(RESPONSES, "cohort.csv")      # id, age, sex, the 151 items (imputed where needed)
MASK_CSV = os.path.join(RESPONSES, "imputed_mask.csv")  # id, one column per item: True where the answer was imputed
ID = "id"

ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2",
            "qwen3-embedding-0.6b"]                     # gte-large-en-v1.5 is the encoder chosen in advance
PRIMARY = ENCODERS[0]
COVARIATES = ["age", "sex"]                             # every score is residualized on them
PCTILE = 95.0                                           # an outlier: distance above the grouping's own 95th percentile
FEMALE, MALE = 0.0, 1.0                                 # sex at birth as step 1 codes it
Z = 1.959964                                            # the normal 97.5th percentile
MIN_NIGHTS = 14                                         # Fitbit nights a person needs for the sleep outcomes


def work(*parts):
    """A path in work/, creating the folder."""
    os.makedirs(os.path.join(WORK, *parts[:-1]), exist_ok=True)
    return os.path.join(WORK, *parts)


def download_dir(name):
    """downloads/<name>/, created."""
    d = os.path.join(DOWNLOADS, name)
    os.makedirs(d, exist_ok=True)
    return d
