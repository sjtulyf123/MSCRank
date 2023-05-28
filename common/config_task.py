"""
configure.
"""

REGRESSION_TASK = False
default_pd_ds = "pd"

FILE_SUFFIX = ".pkl"
chunk_size = 1e6

RAW_FILE_PATTERN = "merge_*.txt"
BEHAVIOR_MAP_PREFIX = "app_id_udf_map_"



LABEL_HIT = (1, 11)

STEP_LEN = 3  # 5
SESSION_LEN_MIN = 6  # 10
SESSION_LEN_MAX = 100

SESSION_TOP_K = 30

BEHAVIOR_CNT_MAX = 100

# this works as bias term, set to None if not using.
ADD_COLUMN = None

SHARE_INDEX = None

# DATA PARAMETER
NEG_SAMP_RATIO = 1  # negative samples' sampling ratio, (0,1]

FLTR = 20  # threshold for building feature index
APPID_FLTR = 20  # threshold for building app.id feature index
APP_CNT_THRESH = 1  # threshold for filtering app candidate

DELIMITER = '\t'  # specify delimiter in provided raw data file.

MULTI_HOT_DELIMITER = ','


if __name__ == '__main__':
    pass
