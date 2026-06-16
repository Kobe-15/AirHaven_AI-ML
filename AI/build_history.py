# build_history.py  ← run this ONCE before the exhibit

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from data_loader import init_firebase, fetch_and_average
from config import HISTORY_PATH

# Temporarily override to use all 3 nodes
import config
config.PROTOTYPE_KEYS = ['node_1', 'node_2', 'node_3']
config.FIELD_NODES    = ['node_1', 'node_2', 'node_3']

init_firebase()
print("Fetching all 3 nodes for history...")
area_df = fetch_and_average()

os.makedirs('data', exist_ok=True)
save_df = area_df.copy()
save_df.index = save_df.index.tz_localize(None)
save_df.to_csv(HISTORY_PATH)
print(f"History saved: {len(area_df)} rows → {HISTORY_PATH}")