from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

raw_data_path = project_root / "data" / "raw" / "car-price-data.csv"


igr_flag_multiplier = 1.5 
igr_remove_multiplier = 3.0  
zscore_threshhold = 3.0

