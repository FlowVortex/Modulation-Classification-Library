import torch
from torch import nn


class Model(nn.Module):
    """`Connectionist Temporal Classification <https://www.cs.toronto.edu/~graves/icml_2006.pdf>`_
    Args:
        num_classes (int): Number of target labels (excluding blank).
    """

    def __init__(self, configs) -> None:
        super(Model, self).__init__()

        self.d_model = configs.d_model
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout
        self.n_classes = configs.n_classes

        # 双向 LSTM
        self.backbone = nn.LSTM(
            input_size=2,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.n_layers > 1 else 0,
        )

        self.classifier = nn.Linear(2 * self.d_model, self.n_classes)
        # CTC   n_classes+1
        # self.classifier = nn.Linear(2 * self.d_model, self.n_classes + 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x: (Batch, Channels, Length)

        # (B, C, L) -> (B, L, C)
        x = torch.transpose(x, 1, 2)

        # rnn_out: (Batch, Length, 2 * d_model)
        rnn_out, _ = self.backbone(x)

        # logits: (Batch, Length, n_classes + 1)
        # logits = self.classifier(rnn_out)

        # CTC Loss
        # log_probs = nn.functional.log_softmax(logits, dim=2)
        # log_probs = log_probs.transpose(0, 1)
        # return log_probs

        last_output = rnn_out[:, -1, :]

        return self.classifier(last_output)
