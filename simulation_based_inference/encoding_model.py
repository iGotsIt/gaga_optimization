import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

## We can try different encoding methods: encoding the summary statistics directly, encoding the raw data,
## or using a "summary network" to get summaries (which I think is just encoding raw data but not sure)

## we can test to see if it can distinguish between 2 different experimental conditions?

## we may have to do a bit of information leakage -- it should be padded to the max length of the sequence across test and train sets,
## or we just use adaptivemaxpool and batch-level padding


class ConvNet(torch.nn.Module):
    """
    Convolutional neural network for 1D time series
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, output_dim, padding=1
    ):
        super().__init__()
        conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        relu1 = torch.nn.ReLU()
        maxpool1 = torch.nn.MaxPool1d(kernel_size=2)

        conv2 = torch.nn.Conv1d(
            out_channels, out_channels * 2, kernel_size, stride, padding
        )
        relu2 = torch.nn.ReLU()
        maxpool2 = torch.nn.MaxPool1d(kernel_size=2)

        conv3 = torch.nn.Conv1d(
            out_channels * 2, output_dim, kernel_size, stride, padding
        )
        relu3 = torch.nn.ReLU()

        adaptive_pool = torch.nn.AdaptiveMaxPool1d(
            1
        )  # Adaptive pooling to get a fixed-size output --> (batch, output_dim, 1)
        flatten = torch.nn.Flatten()  # Flatten to (batch, output_dim)

        self.model = torch.nn.Sequential(
            conv1,
            relu1,
            maxpool1,
            conv2,
            relu2,
            maxpool2,
            conv3,
            relu3,
            adaptive_pool,
            flatten,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class SimpleClassifier(torch.nn.Module):
    """
    Simple classifier for distinguishing between two experimental conditions
    """

    def __init__(self, out_channels, hidden_dim, output_dim):
        super().__init__()
        self.conv_net = ConvNet(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            output_dim=hidden_dim,
            padding=1,
        )
        self.mlp = MLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        self.classifier = torch.nn.Linear(output_dim, 2)

    def forward(self, x):
        x = self.conv_net(x)
        x = self.mlp(x)
        x = self.classifier(x)
        return x


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Custom dataset for time series data
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float(), torch.tensor(
            self.labels[index]
        ).long()


class TimeSeriesDataloader(torch.utils.data.DataLoader):
    """
    Custom dataloader for time series data
    """

    def __init__(self, dataset, batch_size=32, shuffle=True):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
        )


def collate(batch):
    """
    Custom collate function to pad sequences
    """
    features, labels = zip(*batch)

    transposed_features = [f.transpose(0, 1) for f in features]  # (3, T) -> (T, 3)

    data = torch.nn.utils.rnn.pad_sequence(
        transposed_features, batch_first=True
    )  ## (B, T_max, 3)
    labels = torch.tensor(labels)
    data = data.transpose(1, 2)  # Transpose --> (B, 3, T_max) for Conv1d input
    return data, labels


def train(dataset, epochs):

    classifier = SimpleClassifier(out_channels=16, hidden_dim=32, output_dim=16)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    dataloader = TimeSeriesDataloader(dataset, batch_size=32, shuffle=True)
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        n_samples = 0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            outputs = classifier(x_batch)
            loss_value = loss(outputs, y_batch)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        epoch_loss = running_loss / n_samples
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return classifier, epoch_losses


def test(dataset, classifier):
    dataloader = TimeSeriesDataloader(dataset, batch_size=32, shuffle=False)

    predictions = []
    true_labels = []
    classifier.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            logits = classifier(x_batch)  # (B, 2)
            predictions.append(logits.argmax(dim=1))  # (B,)
            true_labels.append(y_batch)

    predictions = torch.cat(predictions).numpy()
    true_labels = torch.cat(true_labels).numpy()
    accuracy = (predictions == true_labels).mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, predictions, true_labels


def k_fold_cross_validation(dataset, k=5, epochs=10, seed=0):
    """
    Stratified k-fold CV. Returns per-fold accuracies, per-fold confusion matrices
    (rows = true label, cols = predicted label), and the confusion matrix summed over folds.
    """
    labels = np.array(dataset.labels).astype(int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    accuracies = []
    confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold + 1}/{k}")
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        classifier, _ = train(train_dataset, epochs)
        accuracy, predictions, true_labels = test(val_dataset, classifier)
        accuracies.append(accuracy)
        confusion_matrices.append(
            confusion_matrix(true_labels, predictions, labels=[0, 1])
        )

    total_confusion_matrix = np.sum(confusion_matrices, axis=0)
    print(f"Accuracy over {k} folds: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    return accuracies, confusion_matrices, total_confusion_matrix
