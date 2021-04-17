from torch.utils.data import Dataset, DataLoader
from wavenet.dataset import DatasetConfig
from wavenet.model import WaveNetKWS
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import torch


class WaveNetDataset(Dataset):
    def __init__(self, path: Path, input_size: int):
        self.input_size = input_size
        with open(path / "data.pkl", "rb") as file:
            data = pickle.load(file)
            self.config = DatasetConfig(**data["config"])
            self.items = data["items"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {
            "inputs": item["features"][:, -self.input_size :],
            "labels": item["label"],
            "ids": index,
        }


@torch.no_grad()
def validate(model, loader, device):
    total = tp = tn = fp = fn = 0
    wrong_ids = []
    for batch in tqdm(loader, leave=False, desc="Validation"):
        inputs = batch["inputs"].to(device)
        outputs = model.forward(inputs)
        predicted = torch.argmax(outputs, dim=1).cpu().detach()
        flat_pred = predicted.numpy().flatten().tolist()
        flat_labels = batch["labels"].numpy().flatten().tolist()
        for p, l in zip(flat_pred, flat_labels):
            if p == 0 and l == 0:
                tn += 1
            if p == 1 and l == 1:
                tp += 1
            if p == 0 and l == 1:
                fn += 1
            if p == 1 and l == 0:
                fp += 1
        for i in range(len(batch["ids"])):
            if flat_pred[i] != flat_labels[i]:
                wrong_ids.append(batch["ids"][i])
        total += len(flat_labels)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    counts = (tp, tn, fp, fn)
    wrong_ids = list(map(int, wrong_ids))
    return precision, recall, counts, wrong_ids


def train_model(
    model,
    device,
    train_loader,
    test_loader,
    criterion=None,
    optimizer=None,
    start_epoch=1,
    end_epoch=5,
):
    criterion = criterion or torch.nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
    epoch = start_epoch

    while epoch <= end_epoch:
        train_loss = 0
        for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            with torch.set_grad_enabled(True):
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.cpu().detach().item()
                loss.backward()
                optimizer.step()

        train_loss /= len(train_loader)
        precision, recall, counts, wrong_ids = validate(model, test_loader, device)
        yield dict(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            precision=precision,
            recall=recall,
            counts=counts,
            wrong_ids=wrong_ids,
        )
        epoch += 1


def model_from_checkpoint(path: Path, device=torch.device("cpu")):
    checkpoint = torch.load(path, map_location=device)
    model = WaveNetKWS(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset_config = checkpoint["dataset_config"]
    dataset_config = DatasetConfig(**dataset_config)
    return model, dataset_config


if __name__ == "__main__":
    DATASET_PATH = Path("../datasets/v1")
    SESSION_PATH = Path("../checkpoints/v1")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    END_EPOCH = 5

    MODEL_CONFIG = dict(
        num_groups=10,
        group_size=4,
        kernel_size=2,
        input_channels=40,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=64,
        out_hidden_channels=128,
        out_classes=2,
    )
    model = WaveNetKWS(**MODEL_CONFIG)
    model = model.eval()
    model = model.to(DEVICE)

    dataset = WaveNetDataset(DATASET_PATH, model.input_size)
    split_sizes = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, split_sizes)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    it = train_model(
        model=model,
        device=DEVICE,
        train_loader=train_loader,
        test_loader=test_loader,
        end_epoch=END_EPOCH,
    )

    print("Training started")
    for status in it:
        print(f"=== Epoch {status['epoch']} report ===")
        print(f"- train_loss: {status['train_loss']}")
        print(f"- precision: {status['precision']}")
        print(f"- recall: {status['recall']}")
        tp, tn, fp, fn = status["counts"]
        print(f"- counts: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"- wrong ids: {status['wrong_ids'][:20]}")

        checkpoint = dict(
            dataset_config=dataset.config.asdict(),
            model_config=MODEL_CONFIG,
            epoch=status["epoch"],
            train_loss=status["train_loss"],
            precision=status["precision"],
            recall=status["recall"],
            model_state_dict=model.state_dict(),
            optimizer_state_dict=status["optimizer"].state_dict(),
        )
        path = SESSION_PATH.with_suffix(f".epoch{status['epoch']}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved -> {path}")
