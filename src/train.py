import os
import pandas as pd
import torch

from . import config
from . import dataset
from . import dispatcher
from . import model

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))

MODEL = os.environ.get("MODEL")
EPOCHS = int(os.environ.get("EPOCHS"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
LR = float(os.environ.get("LR"))

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


def train():
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df["excerpt"]
    valid_df = valid_df["excerpt"]

    train_data = dataset.TextDataset(train_df.to_list(), ytrain)
    valid_data = dataset.TextDataset(valid_df.to_list(), yvalid)

    # data is ready to train
    predictor = model.CommonLitBertBaseModel(
        model=dispatcher.MODELS[MODEL], device=config.DEVICE
    )
    predictor.to(config.DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=LR)
    predictor.fit(
        train_data,
        optimizer,
        criterion,
        val_set=valid_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_path=f"models/{MODEL}_{FOLD}",
        shuffle=False,
    )


if __name__ == "__main__":
    train()
