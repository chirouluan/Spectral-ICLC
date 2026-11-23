"""LUMIR classification architectures; original definitions, no legacy task pipelines."""
import torch
import torch.nn as nn

class CNN1DClassifier(nn.Module):

    def __init__(self, seq_len, n_classes, in_channels=1, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=2), nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.conv(x)
        out = self.classifier(out)
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]

class TransformerClassifier(nn.Module):

    def __init__(self, seq_len, n_classes, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits
