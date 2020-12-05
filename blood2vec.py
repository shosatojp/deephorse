import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Blood2Vec(torch.nn.Module):
    '''
    embedding by blood relationship
    '''
    def __init__(self, horse_count: int, ndim: int) -> None:
        super().__init__()

        self.horse_count = horse_count
        self.ndim = ndim

        hidden_dim = 10
        self.embed = torch.nn.Embedding(horse_count, ndim)
        self.fc1 = torch.nn.Linear(ndim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, horse_count)

    def forward(self, x):
        out = self.embed(x).reshape((-1, self.ndim * 2))
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = torch.log_softmax(out, dim=0)
        return out


class HorsesDataset(Dataset):
    def __init__(self, csvfile) -> None:
        self.df = pd.read_csv(csvfile)
        self.len = self.df.shape[0]
        self.availables = self.df.loc[(self.df['ancestor_1'] != -1) & (self.df['ancestor_2'] != -1)].index
        print(f'number of available horses is {len(self.availables)}')

    def __getitem__(self, index: int):
        row = self.df.loc[self.availables[index]]
        me = row[0]
        father = row['ancestor_1']
        mother = row['ancestor_2']
        return torch.tensor([father, mother]), me

    def __len__(self) -> int:
        return self.len


if __name__ == "__main__":

    epochs = 100
    batch_size = 150

    dataset = HorsesDataset('horses.pedigree.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    print(len(dataset), dataset[0])

    device = 'cuda'

    net = Blood2Vec(len(dataset), 10)
    net.to(device)

    lossfn = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        print(epoch)
        total_loss = .0
        for inputs, targets in tqdm(dataloader):

            optim.zero_grad()

            out = net(inputs.to(device))
            one_hot = torch.nn.functional.one_hot(targets, num_classes=len(dataset))
            loss = lossfn(out, one_hot.float().to(device))

            loss.backward()
            optim.step()

            total_loss += loss.item()
        print(total_loss)
