import os
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

class Group_DataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers=0):
        print('######[DEBUG]: Using Group Dataloader')
        self.dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers
        )

class Weight_DataLoader:
    def __init__(self, dataset, batch_size, loss_file, init_value=1.0, num_workers=0, epsilon=0.2):
        print('######[DEBUG]: Using Weight Dataloader')

        self.epsilon = epsilon
        self.weight_decay = 0.8

        self.loss_weight = {}
        if os.path.exists(loss_file):
            print('######[DEBUG]: Load Loss File From %s'%loss_file)
            with open(loss_file, 'r', encoding='utf8') as f:
                raw_data = json.load(f)
                for file in raw_data:
                    self.loss_weight[file] = raw_data[file]
        else:
            print('######[DEBUG]: No Loss File')

        for key in dataset.get_names():
            if key not in self.loss_weight.keys():
                self.loss_weight[key] = init_value

        num_samples = int(math.ceil(len(dataset) / batch_size) * batch_size)
        self.weightSampler = WeightedRandomSampler(list(self.loss_weight.values()), num_samples=num_samples)

        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=self.weightSampler, num_workers=num_workers
        )

    def update_dict(self, loss, keys):
        for tag in keys:
            self.loss_weight[tag] = self.loss_weight[tag]*self.weight_decay + loss*(1-self.weight_decay)

    def update_weight(self):
        if np.random.random() > self.epsilon:
            print('######[DEBUG]: Weight Update')
            weight = np.array(list(self.loss_weight.values()))
            update_weight = weight / weight.sum()
            update_weight = np.maximum(update_weight, 1e-6)
            self.weightSampler.weights = torch.as_tensor(update_weight, dtype=torch.double)
        else:
            print('######[DEBUG]: Random Update')
            weight = np.ones(len(self.loss_weight))
            self.weightSampler.weights = torch.as_tensor(weight, dtype=torch.double)

    def save_weight(self, output_file):
        json_str = json.dumps(self.loss_weight, sort_keys=True, indent=4, separators=(',', ': '))
        with open(output_file, 'w') as f:
            f.write(json_str)

if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    from models.restormer.dataset_utils import Dataset_PairedImage

    config_path = '/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/restormer/test_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    dataset = Dataset_PairedImage(
        img_dir='/home/quan/Desktop/tempary/temp/good',
        config=config['dataset']
    )

    dataLoader = Group_DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    for i, data_batch in enumerate(dataLoader.dataloader):
        noise_images, gt_images, names = data_batch
        # print(noise_images.shape, gt_images.shape)
        # print(names)
        # print(data_batch)

        noise_img = noise_images.numpy()[0, ...]
        gt_img = gt_images.numpy()[0, ...]
        noise_img = np.transpose(noise_img, (1, 2, 0))
        gt_img = np.transpose(gt_img, (1, 2, 0))

        plt.figure('noise')
        plt.imshow(noise_img)
        plt.figure('gt')
        plt.imshow(gt_img)
        plt.show()