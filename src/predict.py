import os
import pandas as pd
import numpy as np
import torch

from . import config
from . import dispatcher
from . import model

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict(model_path):
    df = pd.read_csv(TEST_DATA)
    df = df[["id", "excerpt"]]
    test_idx = df["id"].values
    test_excerpts = tuple(df["excerpt"].to_list())
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        predictor = model.CommonLitBertBaseModel(
            dispatcher.MODELS[MODEL], config.DEVICE
        )
        predictor.load_state_dict(torch.load(f"{model_path}/{MODEL}_{FOLD}_5.pt"))
        preds = predictor.predict(test_excerpts)

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(
        np.column_stack((test_idx, predictions)), columns=["id", "target"]
    )
    return sub


if __name__ == "__main__":
    submission = predict(model_path="models/")
    submission.to_csv(f"results/{MODEL}_submission.csv", index=False)
