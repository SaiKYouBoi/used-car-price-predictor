import pandas as pd
import config

def load_data(path=config.raw_data_path):
    df = pd.read_csv(path)
    return df


def check_structure(df: pd.DataFrame):
    print("\nStructure check:")
    print(f"\nnumber of rows {df.shape[0]}, number of columns df.shape[1]")
    print(f"\nColumn dtypes")
    print(f"{df.dtypes}")
    print("\nfirst 5 rows:")
    print(f"{df.head()}")
    print("\nmissing values:")
    print(f"{df.isna().sum()}")
    print("\nduplicate rows")
    print(f"{df.duplicated().sum()}")


if __name__ == "__main__":
    data = load_data()
    check_structure(data)
