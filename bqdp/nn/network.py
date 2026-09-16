# bqdp/nn/network.py
import torch
from torch import nn





class SequenceRecommender(nn.Module):
    def __init__(self, table_size: int, embedding_dim: int, num_layers: int, num_heads: int, feedforward_dim: int, dropout: float, max_sequence_length: int):
        super().__init__()
        self.table_size = table_size
        self.max_sequence_length = max_sequence_length
        self.embedding_scale = embedding_dim**0.5

        self.item_embedding = nn.Embedding(table_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model = embedding_dim,
                    nhead = num_heads,
                    dim_feedforward = feedforward_dim,
                    dropout = dropout,
                    activation = "gelu",
                    batch_first = True,
                    norm_first = True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        
        self._reset_parameters()


    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()


    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        length = sequences.size(1)
        positions = torch.arange(length, device=sequences.device)

        hidden = self.item_embedding(sequences) * self.embedding_scale
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_dropout(hidden)

        # padded steps are zeroed after every block so they contribute no value mass
        valid = (sequences != 0).unsqueeze(-1).to(hidden.dtype)
        causal = self._causal_mask(length, hidden.device, hidden.dtype)

        hidden = hidden * valid
        
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal)
            hidden = hidden * valid
        
        return self.output_norm(hidden)


    def full_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.item_embedding.weight.t())


    @staticmethod
    def _causal_mask(length: int, device, dtype) -> torch.Tensor:
        mask = torch.full((length, length), float("-inf"), device=device, dtype=dtype)
        
        return torch.triu(mask, diagonal=1)