from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

raw_data_path = project_root / "data" / "raw" / "car-price-data.csv"


igr_flag_multiplier = 1.5 
igr_remove_multiplier = 3.0  
zscore_threshold = 3.0

current_year = 2026
min_plausible_year = 1990


owner_mapping = {
    "Test Drive Car":0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4
    }