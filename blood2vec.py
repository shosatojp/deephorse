import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
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

        self.fc1 = torch.nn.Linear(6*ndim, ndim)

    def forward(self, x, target_id):
        out = self.embed(x)
        # out = out.sum(dim = 1)

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
                                           self.df.columns))[:6]
        self.availables = self.df.loc[
            functools.reduce(lambda a, b: a & b,
                             map(lambda label:self.df[label] != -1, self.ancestor_labels))
        ].index.to_numpy()

        self.context = torch.tensor(np.array(self.df[self.ancestor_labels].loc[self.availables]))

        print(f'number of available horses is {len(self.availables)}')

    def __getitem__(self, index: int):
        me = self.availables[index]
        context = self.context[index]
        return context, me

    def __len__(self) -> int:
        return len(self.availables)


if __name__ == "__main__":
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    epochs = 20
    batch_size = 1000
    latent_size = 30

    dataset = HorsesDataset('horses.pedigree.csv')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=os.cpu_count())

    device = 'cuda'
    # device = 'cpu'

    # figure
    fig = plt.figure(figsize=(12., 6.))
    fig.suptitle('loss and accuracy')
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_acc = fig.add_subplot(1, 2, 2)
    ax_loss.set_title('loss')
    ax_loss.set_xlabel('epochs')
    ax_loss.set_ylabel('l1 loss')
    ax_acc.set_title('accuracy')
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_xlabel('epochs')

    loss_history, acc_history = [], []

    net = Blood2Vec(dataset.size, latent_size)

    # === test ===
    test = True
    if test:
        net.load_state_dict(torch.load(os.path.join(checkpoints_dir, '00019.pkl')))
        target = net.get_latent(torch.tensor([0]))
        latent_table = net.get_latent(torch.tensor(dataset.availables))

        # 類似度
        diff = latent_table - target.expand(latent_table.shape)
        print(diff.shape)
        mul = diff * diff
        mul = torch.sum(mul, dim=1)

        print(mul.shape)

        root = torch.sqrt(mul)

        arg = torch.argsort(root, dim=0)
        print(dataset.df.loc[dataset.availables[arg]])
        print(arg[:10])
        print(root[arg[:10]])

        exit(0)

    # === train ===
    net.to(device)

    # number of negative sample
    neg_count = 5
    pos_count = neg_count

    lossfn = torch.nn.L1Loss()
    optim = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(1, 1 + epochs):
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
                              targets.view((*targets.shape, 1)).expand((*targets.shape, pos_count + neg_count)))
                pn_targets = pn_targets.reshape((-1,))
                pn_answers = pn_answers.reshape((-1,))

                pn_inputs = inputs.view((*inputs.shape, 1)).expand((*inputs.shape, pos_count + neg_count))
                pn_inputs = pn_inputs.transpose(1, 2).reshape((-1, inputs.shape[1]))

                # randomise
                inputs_targets_answers = torch.cat([pn_inputs, pn_targets.view(-1, 1), pn_answers.view(-1, 1).long()], dim=1)
                inputs_targets_answers = inputs_targets_answers[torch.randperm(inputs_targets_answers.shape[0])]
                pn_inputs = inputs_targets_answers[:, :-2]
                pn_targets = inputs_targets_answers[:, -2]
                pn_answers = inputs_targets_answers[:, -1]

                with torch.set_grad_enabled(phase == 'train'):

                    out = net(pn_inputs.to(device), pn_targets.to(device))
                    loss = lossfn(out, pn_answers.float().to(device))
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
            print(f'epoch = {epoch}, phase = {phase}, loss = {loss_per_input: .2g}, acc = {acc_per_input: .2g}')
            ax_loss.plot(np.array(loss_history), color='red')
            ax_acc.plot(np.array(acc_history), color='green')
            plt.draw()
            plt.pause(0.01)
        # print(out)

        if (epoch) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(checkpoints_dir, f'{epoch:05d}.pkl'))
    plt.savefig('loss_acc.png')
