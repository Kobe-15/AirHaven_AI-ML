TARGETS = ['humidity', 'temperature', 'NO2', 'O3', 'PM10', 'PM2_5', 'co']

SENSOR_TARGETS = ['humidity', 'temperature', 'NO2', 'O3', 'PM10', 'PM2_5', 'co']

PROTOTYPE_KEYS = ['node_1', 'node_2', 'node_3']

FIELD_NODES = ['node_1', 'node_2', 'node_3']

FIELD_MAP = {
    'humidity':    'humidity',
    'temperature': 'temperature',
    'no2':         'NO2',
    'o3_ppb':      'O3',
    'pm100':       'PM10',
    'pm25':        'PM2_5',
    'co':          'co',
}

HOURLY_MODEL_PATH  = 'models/hourly_model.joblib'
HOURLY_SCALER_PATH = 'models/hourly_scaler.joblib'
HOURLY_META_PATH   = 'models/hourly_meta.json'

DAILY_MODEL_PATH   = 'models/daily_model.joblib'
DAILY_SCALER_PATH  = 'models/daily_scaler.joblib'
DAILY_META_PATH    = 'models/daily_meta.json'

HISTORY_PATH = 'data/area_history.csv'

HOURLY_STEPS = 24
DAILY_STEPS  = 6