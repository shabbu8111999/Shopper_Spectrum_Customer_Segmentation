
import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding="latin1")
        return df
    except Exception as e:
        raise Exception(f"Dataset could not be loaded: {e}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")