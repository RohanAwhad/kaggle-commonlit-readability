import optuna
import torch
from tqdm.notebook import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class CommonLitBertModel(torch.nn.Module):
    def __init__(self, 
                 model_path=None, 
                 tokenizer_path=None, 
                 config_path=None, 
                 device="cpu",
                 max_len=512,
                 dropout_rate=0
        ):
        super(CommonLitBertModel, self).__init__()
        self.device = device
        self.max_len = max_len
        
        if model_path is not None:
            self.config = AutoConfig.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.pre_trained = AutoModel.from_pretrained(model_path)
        else:
            self.config = AutoConfig.from_pretrained(config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=self.config)
            self.pre_trained = AutoModel.from_config(self.config)
            
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, text):
        inp = self.tokenizer(
            text, return_tensors="pt", padding='longest', truncation=True, max_length=self.max_len
        ).to(self.device)
        x = self.pre_trained(**inp).pooler_output
        x = self.dropout(x)
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
        val_batch_size=None,
        model_path="model",
        shuffle=False,
        patience=2,
        scheduler=None,
        save_model=True,
        trial=None
    ):

        train_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle
        )

        if val_set is not None:
            val_batch_size = batch_size if val_batch_size is None else val_batch_size
            val_dataloader = torch.utils.data.DataLoader(
                val_set, batch_size=val_batch_size, shuffle=shuffle
            )

        prev_val_loss = None
        for epoch in range(1, epochs + 1):
            progress_bar = tqdm(
                total=len(train_dataloader) + len(val_dataloader),
                desc=f"Epoch {epoch}/{epochs}",
                leave=True,
            )
            # Training
            train_loss = []
            optimizer.zero_grad()
            self.train()
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
            self.eval()
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
         
            progress_bar.close()
            val_loss = sum(val_loss) / len(val_loss)
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if scheduler is not None:
                scheduler.step(val_loss)

            if prev_val_loss is None or prev_val_loss > val_loss:
                prev_val_loss = val_loss
                countdown_patience = patience
                print(f'Min val loss: {prev_val_loss} at Epoch: {epoch}')
                if save_model:
                    torch.save(self.state_dict(), f"{model_path}.pt")
            else:
                if countdown_patience == 0:
                    break
                else:
                    countdown_patience -= 1

        if save_model:
            self.tokenizer.save_pretrained(f'tokenizers/{MODEL}')
            self.config.save_pretrained(f'configs/{MODEL}')

        return prev_val_loss

    def evaluate(self, data, criterion):
        text = data["text"]
        target = data["target"].to(self.device)
        y_pred = self.forward(text)
        # return criterion(y_pred, target)
        return torch.sqrt(criterion(y_pred, target))

    def predict(self, data_seq, batch_size=16):
        self.eval()
        with torch.no_grad():
            predictions = torch.empty((0, 1)).cpu()
            for i in tqdm(range(0, len(data_seq), batch_size), leave=False):
                pred = self.forward(list(data_seq[i : i + batch_size])).cpu()
                predictions = torch.cat([predictions, pred], axis=0)

        return predictions
