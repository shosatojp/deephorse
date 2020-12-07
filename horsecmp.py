import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import functools
import matplotlib.pyplot as plt


class HorseComp(torch.nn.Module):
    def __init__(self, horse_size, horse_latent,
                 owner_size, owner_latent,
                 trainer_size, trainer_latent,
                 jockey_size, jockey_latent,
                 others_size, rnn_hidden_size) -> None:
        super().__init__()

        self.horse_id_embedding = torch.nn.Embedding(horse_size, horse_latent)
        self.owner_id_embedding = torch.nn.Embedding(owner_size, owner_latent)
        self.trainer_id_embedding = torch.nn.Embedding(trainer_size, trainer_latent)
        self.jockey_id_embedding = torch.nn.Embedding(jockey_size, jockey_latent)

        horse_input_size = horse_latent + owner_latent + trainer_latent + others_size

        self.horse_rnn = torch.nn.RNN(horse_input_size, rnn_hidden_size)
        self.jockey_rnn = torch.nn.RNN(jockey_latent, rnn_hidden_size)

        self.horse_storage = torch.zeros((horse_size, rnn_hidden_size))
        # self.jockey_storage = tor

        # self.fc1 = torch.nn.Linear()

    def forward(self, x: torch.Tensor):
        # x: (batch, 2, data)
        horse_ids = x[:, :, 0].long()
        horse_embedding = self.horse_id_embedding(horse_ids)
        owner_embedding = self.horse_id_embedding(x[:, :, 1].long())
        trainer_embedding = self.horse_id_embedding(x[:, :, 2].long())
        others = x[:, :, 3:].float()

        rnn_inputs = torch.cat([horse_embedding, owner_embedding, trainer_embedding, others], dim=2)

        rnn_output, rnn_hidden = self.horse_rnn(rnn_inputs, self.horse_storage[horse_ids])
        print(rnn_output.shape,rnn_hidden.shape)

        self.horse_storage[]

        pass


if __name__ == "__main__":
    net = HorseComp(100, 10, 100, 10, 100, 10, 100, 10, 2, 15)
    x = torch.tensor([
        [
            [1, 2, 3, 50, 50],
            [1, 2, 3, 50, 50],
        ],
    ])
    y = net(x)
    print(y)
