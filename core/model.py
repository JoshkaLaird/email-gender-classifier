import torch
import torch.nn as nn


class EmailCharCNN(nn.Module):
    def __init__(
        self,
        int_vocab_size: int,
        int_embedding_dim: int,
        int_num_classes: int
    ) -> None:
        super().__init__()

        self.obj_embedding = nn.Embedding(
            num_embeddings=int_vocab_size,
            embedding_dim=int_embedding_dim,
            padding_idx=0
        )

        self.obj_conv_3 = nn.Conv1d(
            in_channels=int_embedding_dim,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.obj_conv_5 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            padding=2
        )

        self.obj_relu = nn.ReLU()
        self.obj_dropout = nn.Dropout(p=0.2)
        self.obj_pool = nn.AdaptiveMaxPool1d(1)
        self.obj_classifier = nn.Linear(128, int_num_classes)

    def forward(self, ts_input_ids: torch.Tensor) -> torch.Tensor:
        ts_x = self.obj_embedding(ts_input_ids)
        ts_x = ts_x.transpose(1, 2)

        ts_x = self.obj_conv_3(ts_x)
        ts_x = self.obj_relu(ts_x)

        ts_x = self.obj_conv_5(ts_x)
        ts_x = self.obj_relu(ts_x)

        ts_x = self.obj_pool(ts_x).squeeze(-1)
        ts_x = self.obj_dropout(ts_x)

        ts_logits = self.obj_classifier(ts_x)
        return ts_logits
