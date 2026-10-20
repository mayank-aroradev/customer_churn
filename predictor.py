
import pickle
from pathlib import Path

import __main__

from pipeline_class import PredictorPipeline

# Compatibility for pickle files saved as __main__.PredictorPipeline
__main__.PredictorPipeline = PredictorPipeline

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)


def predict_output(raw_df):
    return model.predict(raw_df)
