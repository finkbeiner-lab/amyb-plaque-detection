import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'] * 2)))

from model_mrcnn import _default_mrcnn_configs, build_default
from features import build_features
from features import transforms as T


collate_fn = lambda a: tuple(zip(*a)) # one-liner, no need to import

dataset_path = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/amy-def/'
dataset = build_features.AmyBDataset(dataset_path, T.Compose([T.ToTensor()]))


def train_one_epoch(model, optimizer, data_loader, device,):
    model.to(device)
    model.train()

    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = dict([(k, v.to(device)) for k, v in targets.items()])

        losses = model.forward(images, targets)
        print(losses)


#
#
#
# indices = torch.randint(0, 1, len(dataset))
# torch.randint()
#
# # split the dataset in train and test set
# torch.manual_seed(seed)
# indices = torch.randperm(len(dataset)).tolist()
# dataset, dataset_test = [torch.utils.data.Subset(dataset, idxs) for idxs in sorted([indices[:split], indices[split:]], key=lambda a: -len(a))]
#
# # define training and validation data loaders
# data_loader, data_loader_test = [torch.utils.data.DataLoader(dataset, batch_size=batch_size if bool(_) else 1, shuffle=bool(_), num_workers=1, collate_fn=collate_fn) for _ in range(0, 2, -1)]
#
#
#
if __name__ == '__main__':
    config = _default_mrcnn_configs().config_dict

    model = build_default(config, im_size=1024)
    train_one_epoch(model, , )
