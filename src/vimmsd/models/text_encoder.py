import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Wrapper PhoBERT: trả về hidden state từng token (cho cross-attention) và vector <s> (cho concat)."""

    def __init__(self, name="vinai/phobert-base", freeze=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.hidden_size = self.model.config.hidden_size
        self.frozen = freeze
        if freeze:
            self.model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()  # encoder đóng băng thì luôn tắt dropout
        return self

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state
        return {"tokens": tokens, "mask": attention_mask.bool(), "pooled": tokens[:, 0]}
