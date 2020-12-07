import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from torchviz import make_dot
import itertools
import functools
import matplotlib.pyplot as plt


class Blood2Vec(torch.nn.Module):
    '''
    embedding by blood relationship
    '''

    def __init__(self, horse_count: int, ndim: int) -> None:
        super().__init__()

        # number of horses (vocabularies)
        self.horse_count = horse_count

        # latent code dim
        self.ndim = ndim

        # 入力側Embedding
        self.embed = torch.nn.Embedding(horse_count, ndim)
        int_range = 0.5 / self.ndim
        self.embed.weight.data.uniform_(-int_range, int_range)

        # 出力側Embedding
        self.embed_out = torch.nn.Embedding(horse_count, ndim)
        self.embed_out.weight.data.uniform_(-0, 0)

        self.fc1 = torch.nn.Linear(2*ndim, ndim)

    def forward(self, x, target_id):
        out = self.embed(x)
        out = out.reshape((out.shape[0], -1))

        out = torch.relu(self.fc1(out))

        target = self.embed_out(target_id)

        a = torch.mul(out, target)
        a = torch.sum(a, dim=1)

        a = torch.sigmoid(a)
        return a

    def get_latent(self, x: torch.Tensor):
        with torch.set_grad_enabled(False):
            return self.embed(x)


class HorsesDataset(Dataset):
    def __init__(self, csvfile) -> None:
        self.df = pd.read_csv(csvfile)
        self.size = self.df.shape[0]
        print('size = ', self.size)

        # 祖先が見つからなかったもの（-1）は使わないけど、Embeddingの次元数には含める
        self.ancestor_labels = list(filter(lambda e: e.startswith('ancestor_'),
                                           self.df.columns))[:2]
        self.availables = self.df.loc[
            functools.reduce(lambda a, b: a & b,
                             map(lambda label:self.df[label] != -1, self.ancestor_labels))
        ].index.to_numpy()

        # prepare data on memory
        data_path = 'data.pkl'
        if os.path.exists(data_path):
            self.data = torch.load(data_path)
        else:
            self.data = {}
            for _id in self.availables:
                row = self.df.loc[_id]
                me = row[0]
                context = torch.tensor(list(map(lambda label: row[label],
                                                self.ancestor_labels)))
                self.data[_id] = context, me
            torch.save(self.data, data_path)

        print(f'number of available horses is {len(self.availables)}')

    def __getitem__(self, index: int):
        return self.data[self.availables[index]]

    def __len__(self) -> int:
        return len(self.availables)


def CachedDataset():
    pass


if __name__ == "__main__":
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    epochs = 20
    batch_size = 1000
    latent_size = 20

    dataset = HorsesDataset('horses.pedigree.csv')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=os.cpu_count())

    device = 'cuda'
    # device = 'cpu'

    # figure
    fig = plt.figure()
    fig.suptitle('loss and acc')
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_acc = fig.add_subplot(1, 2, 2)
    ax_loss.set_title('loss')
    ax_loss.set_xlabel('epochs')
    ax_loss.set_ylabel('loss')
    ax_acc.set_title('acc')
    ax_acc.set_ylabel('acc')
    ax_acc.set_xlabel('epochs')

    loss_history, acc_history = [], []

    net = Blood2Vec(dataset.size, latent_size)
    net.to(device)

    # === test ===
    # net.load_state_dict(torch.load(os.path.join(checkpoints_dir, '00099.pkl')))
    # a = net.get_latent(torch.tensor([0, 1, 2]).to(device))
    # print(a)
    # exit(0)

    # === train ===

    # number of negative sample
    neg_count = 5
    pos_count = neg_count

    lossfn = torch.nn.L1Loss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        for phase in ['train']:
            # for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            total_loss = .0
            total_acc = .0
            total = 0

            inputs: torch.Tensor
            targets: torch.Tensor
            for inputs, targets in tqdm(dataloader):
                optim.zero_grad()

                # negative sampling
                # answers: positive なら target == 1, negative なら target == 0
                neg_targets = torch.tensor(np.random.choice(dataset.availables, (targets.shape[0], neg_count)))
                pos_targets = targets.view((*targets.shape, 1)).expand((*targets.shape, pos_count))
                pn_targets = torch.cat([pos_targets, neg_targets], dim=1)
                pn_answers = (pn_targets ==
                              targets.view((*targets.shape, 1)).expand((*targets.shape, pos_count + neg_count))).float()
                pn_targets = pn_targets.reshape((-1,))
                pn_answers = pn_answers.reshape((-1,))

                pn_inputs = inputs.view((*inputs.shape, 1)).expand((*inputs.shape, pos_count + neg_count))
                pn_inputs = pn_inputs.transpose(1, 2).reshape((-1, inputs.shape[1]))

                with torch.set_grad_enabled(phase == 'train'):

                    out = net(pn_inputs.to(device), pn_targets.to(device))
                    loss = lossfn(out, pn_answers.to(device))
                    # lossが0.5以下なら正解
                    acc = (torch.sum(loss) < 0.5).item() / torch.numel(loss)

                    if phase == 'train':
                        loss.backward()
                        optim.step()

                    total += pn_inputs.shape[0]
                    total_loss += loss.item() * pn_inputs.shape[0]
                    total_acc += acc * pn_inputs.shape[0]

            # logging
            loss_per_input = total_loss / total
            acc_per_input = total_acc / total
            loss_history.append(loss_per_input)
            acc_history.append(acc_per_input)
            print(epoch + 1, phase, loss_per_input, acc_per_input)
            ax_loss.plot(np.array(loss_history), color='red')
            ax_acc.plot(np.array(acc_history), color='green')
            plt.draw()
            plt.pause(0.01)

        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(checkpoints_dir, f'{epoch:05d}.pkl'))
