import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from torchviz import make_dot


class Blood2Vec(torch.nn.Module):
    '''
    embedding by blood relationship
    '''

    def __init__(self, horse_count: int, ndim: int) -> None:
        super().__init__()

        self.horse_count = horse_count
        self.ndim = ndim

        self.embed = torch.nn.Embedding(horse_count, ndim)
        self.embed_out = torch.nn.Embedding(horse_count, ndim)

    def forward(self, x, target_id):
        out = self.embed(x).sum(1)
        target = self.embed_out(target_id)
        a = torch.mul(out, target)
        a = torch.sum(a, dim=1)
        return a


class HorsesDataset(Dataset):
    def __init__(self, csvfile) -> None:
        self.df = pd.read_csv(csvfile)
        self.size = self.df.shape[0]
        print('size = ', self.size)
        self.availables = self.df.loc[
            (self.df['ancestor_1'] != -1) &
            (self.df['ancestor_2'] != -1)
        ].index.to_numpy()
        print(f'number of available horses is {len(self.availables)}')

    def __getitem__(self, index: int):
        row = self.df.loc[self.availables[index]]
        me = row[0]
        father = row['ancestor_1']
        mother = row['ancestor_2']
        return torch.tensor([father, mother]), me

    def __len__(self) -> int:
        return len(self.availables)


if __name__ == "__main__":
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    epochs = 100
    batch_size = 1000

    dataset = HorsesDataset('horses.pedigree.csv')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=os.cpu_count())

    device = 'cuda'

    net = Blood2Vec(dataset.size, 10)
    net.to(device)

    # number of negative sample
    neg_count = 5

    lossfn = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            total_loss = .0
            total = 0

            inputs: torch.Tensor
            targets: torch.Tensor
            for inputs, targets in tqdm(dataloader):
                total += inputs.shape[0]

                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # negative sampling
                    neg_targets = torch.tensor(np.random.choice(dataset.availables, (targets.shape[0], neg_count)))
                    # positive なら target == 1, negative なら target == 0
                    pn_targets = (torch.cat([targets.reshape((-1, 1)), neg_targets], dim=1) ==
                                  targets.view((*targets.shape, 1)).expand((*targets.shape, 1 + neg_count))).float()
                    pn_inputs = inputs.view((*inputs.shape, 1)).expand((*inputs.shape, 1 + neg_count))
                    pn_inputs = pn_inputs.transpose(1, 2).reshape((-1, inputs.shape[1]))
                    pn_targets = pn_targets.reshape((-1, 1))

                    out = net(inputs.to(device), targets.to(device))

                    loss = lossfn(out, torch.ones((inputs.shape[0],)).to(device))

                    if phase == 'train':
                        loss.backward()
                        optim.step()

                    total_loss += loss.item() * inputs.shape[0]

            epoch_loss_rate = total_loss / total
            print(epoch + 1, phase, epoch_loss_rate)

        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(checkpoints_dir, f'{epoch:05d}.pkl'))
