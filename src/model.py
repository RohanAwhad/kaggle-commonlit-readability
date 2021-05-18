import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class CommonLitBertBaseModel(torch.nn.Module):
    def __init__(self, model, device="cpu"):
        super(CommonLitBertBaseModel, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pre_trained = AutoModel.from_pretrained(model)
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, text):
        inp = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        x = self.pre_trained(**inp).pooler_output
        x = self.linear(x)
        return x

    def fit(
        self,
        train_set,
        optimizer,
        criterion,
        val_set=None,
        epochs=1,
        batch_size=1,
        model_path="model",
        shuffle=False,
    ):

        train_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle
        )

        if val_set is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, shuffle=shuffle
            )

        for epoch in range(1, epochs + 1):
            progress_bar = tqdm(
                total=len(train_dataloader) + len(val_dataloader),
                desc=f"Epoch {epoch}/{epochs}",
                leave=True,
            )
            # Training
            train_loss = []
            optimizer.zero_grad()
            for data in train_dataloader:
                loss = self.evaluate(data, criterion)
                loss.backward()
                train_loss.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({"loss": sum(train_loss) / len(train_loss)})
                progress_bar.update()

            # Validation
            val_loss = []
            with torch.no_grad():
                for data in val_dataloader:
                    loss = self.evaluate(data, criterion)
                    val_loss.append(loss.item())
                    progress_bar.set_postfix(
                        {
                            "loss": sum(train_loss) / len(train_loss),
                            "val loss": sum(val_loss) / len(val_loss),
                        }
                    )
                    progress_bar.update()

            torch.save(self.state_dict(), model_path + f"_{epoch}.pt")
            progress_bar.close()

    def evaluate(self, data, criterion):
        text = data["text"]
        target = data["target"].to(self.device)
        y_pred = self.forward(text)
        return criterion(y_pred, target)

    def predict(self, data_seq, batch_size=16):
        with torch.no_grad():
            predictions = torch.empty((0, 1))
            for i in tqdm(range(0, len(data_seq), batch_size), leave=False):
                pred = self.forward(list(data_seq[i : i + batch_size])).cpu()
                predictions = torch.cat([predictions, pred], axis=0)

        return predictions
