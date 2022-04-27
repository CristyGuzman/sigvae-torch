from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader


if __name__ == '__main__':
    dataset = TUDataset(root='/tmp/csolis-pulse/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
