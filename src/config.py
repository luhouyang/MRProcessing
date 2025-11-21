# ================= PROCESSING CONFIGURATION =================

import matplotlib.pyplot as plt

# --- General Settings ---
PROCESSED_ROOT = "data/processed"
USE_CACHE = True  # Set to True to skip processing if output files exist
QNA_LIMIT = None  # Set to None to process all lines
USE_MARKERS = True # Toggle 3D shape markers for QnA

# --- CSV Format Settings ---
# Indices based on C# output: 
# 6-8(Local XYZ), 24(Time)
CSV_XYZ_INDICES = [0, 1, 2]
CSV_TIMESTAMP_INDEX = 24
CSV_ENCODING = 'utf-8-sig'

# --- Constants ---
DEFAULT_HOLOLENS_2_SPATIAL_ERROR = 1.5
DEFAULT_GAUSSIAN_DENOMINATOR = 2 * (DEFAULT_HOLOLENS_2_SPATIAL_ERROR**2)
DEFAULT_TARGET_VOXEL_RESOLUTION = 48
DEFAULT_CMAP = plt.get_cmap('jet')
DEFAULT_BASE_COLOR = [0.0, 0.0, 0.0]

# --- File Extensions ---
MESH_PC_VOXEL_EXTENSION = ".ply"
BACKUP_MODEL_EXTENSION = '.glb'
VOICE_EXTENSION = ".mp3"
SANITY_CHECK_EXTENSION = ".png"
TRANSCRIPT_EXTENSION = ".txt"

# --- Fixation Settings ---
FIXATION_ALGORITHM = 'velocity' 

FIXATION_PARAMS = {
    'velocity_threshold': 1.0,   # Units/sec
    'dispersion_threshold': 0.1, # Units
    'duration_threshold': 100,   # ms
    'min_points': 2
}

# --- Model ID Mapping ---
# Maps Folder Name -> ID
ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1', 'FH0008': '2', 'IN0003': '3', 'IN0008': '4', 'IN0009': '5',
    'IN0017': '6', 'IN0081': '7', 'IN0104': '8', 'IN0135': '9', 'IN0148': '10',
    'IN0220': '11', 'IN0228': '12', 'IN0232': '13', 'IN0239': '14', 'IN0277': '15',
    'MY0001': '16', 'MY0002': '17', 'MY0004': '18', 'MY0006': '19', 'MY0007': '20',
    'ND0001': '21', 'NM0001': '22', 'NM0002': '23', 'NM0009': '24', 'NM0010': '25',
    'NM0014': '26', 'NM0015': '27', 'NM0017': '28', 'NM0041': '29', 'NM0049': '30',
    'NM0066': '31', 'NM0070': '32', 'NM0072': '33', 'NM0073': '34', 'NM0079': '35',
    'NM0080': '36', 'NM0099': '37', 'NM0106': '38', 'NM0133': '39', 'NM0135': '40',
    'NM0144': '41', 'NM0154': '42', 'NM0156': '43', 'NM0159': '44', 'NM0168': '45',
    'NM0173': '46', 'NM0175': '47', 'NM0189': '48', 'NM0191': '49', 'NM0206': '50',
    'SB0002': '51', 'SB0004': '52', 'SI0001': '53', 'SJ0503': '54', 'SJ0504': '55',
    'SK0001': '56', 'SK0002': '57', 'SK0003': '58', 'SK0004': '59', 'SK0005': '60',
    'SK0013': '61', 'SS0001': '62', 'TJ0004': '63', 'TJ0005': '64', 'TJ0010': '65',
    'TK0002': '66', 'TK0048': '67', 'TK0057': '68', 'UD0001': '69', 'UD0003': '70',
    'UD0005': '71', 'UD0006': '72', 'UD0011': '73', 'UD0013': '74', 'UD0014': '75',
    'UD0016': '76', 'UD0023': '77', 'UD0302': '78', 'UD0304': '79', 'UD0308': '80',
    'UD0318': '81', 'UD0322': '82', 'UD0411': '83', 'UD0412': '84', 'UK0001': '85',
    'IN0295': '86', 'IN0306': '87', 'MH0037': '88', 'NM0239': '89', 'NZ0001': '90',
    'SK0035': '91', 'TK0020': '92', 'UD0028': '93', 'rembak7': 'A',
}

# --- Colors ---
DEFAULT_QNA_ANSWER_COLOR_MAP = {
    "面白い・気になる形だ": {"rgb": [0, 255, 255], "name": "cyan"},
    "美しい・芸術的だ": {"rgb": [0, 255, 0], "name": "green"},
    "不思議・意味不明": {"rgb": [255, 255, 0], "name": "yellow"},
    "不気味・不安・怖い": {"rgb": [255, 0, 0], "name": "red"},
    "何も感じない": {"rgb": [128, 128, 128], "name": "grey"},
    "Interesting and attentional shape": {"rgb": [0, 255, 255], "name": "cyan"},
    "Beautiful and artistic": {"rgb": [0, 255, 0], "name": "green"},
    "Strange and incomprehensible": {"rgb": [255, 255, 0], "name": "yellow"},
    "Creepy / unsettling / scary": {"rgb": [255, 0, 0], "name": "red"},
    "Feel nothing": {"rgb": [128, 128, 128], "name": "grey"},
}

# --- Universal Localization Maps ---
UNIVERSAL_LABEL_MAP = {
    "面白い・気になる形だ": "面白い",
    "美しい・芸術的だ": "美しい",
    "不思議・意味不明": "不思議",
    "不気味・不安・怖い": "怖い",
    "何も感じない": "何も感じない",
    "NO RESPONSE": "NO RESPONSE",
    "Interesting and attentional shape": "Interesting",
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange",
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing",
}

UNIVERSAL_SYMBOL_MAP = {
    "面白い・気になる形だ": "◇",
    "美しい・芸術的だ": "□",
    "不思議・意味不明": "△",
    "不気味・不安・怖い": "X",
    "何も感じない": "○",
    "NO RESPONSE": "・",
    "面白い": "◇", "美しい": "□", "不思議": "△", "怖い": "X",
    "Interesting and attentional shape": "◇",
    "Beautiful and artistic": "□",
    "Strange and incomprehensible": "△",
    "Creepy / unsettling / scary": "X",
    "Feel nothing": "○",
    "Interesting": "◇", "Beautiful": "□", "Strange": "△", "Scary": "X"
}

# --- 3D Shape Mapping ---
QNA_SHAPE_KEY_MAP = {
    "面白い・気になる形だ": 'diamond',
    "美しい・芸術的だ": 'square',
    "不思議・意味不明": 'triangle',
    "不気味・不安・怖い": 'x',
    "何も感じない": 'dot',
    "Interesting and attentional shape": 'diamond',
    "Beautiful and artistic": 'square',
    "Strange and incomprehensible": 'triangle',
    "Creepy / unsettling / scary": 'x',
    "Feel nothing": 'dot',
}