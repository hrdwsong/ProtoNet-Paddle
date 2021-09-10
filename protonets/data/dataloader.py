import paddle
from paddle.io import DataLoader

from .sampler import CategoriesSampler
from .dataset import GeneralDataset

from paddle.vision.transforms import Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, \
    Normalize, Compose

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_dataloader(config, mode='train'):
    if mode == 'train':
        trfms = Compose([
            Resize(84),  # 把短边按比例缩放到84
            CenterCrop((84, 84)),  # 中心裁剪到(84,84).相比直接resize，这样做避免了形变。
            ToTensor(),
            Normalize(MEAN, STD),
        ])
    else:
        trfms = Compose([
            Resize(84),
            CenterCrop((84, 84)),
            ToTensor(),
            Normalize(MEAN, STD)
        ])

    dataset = GeneralDataset(data_root=config['data_root'], mode=mode, trfms=trfms)

    sampler = CategoriesSampler(label_list=dataset.label_list,
                                label_num=dataset.label_num,
                                episode_size=1,
                                episode_num=config['data.train_episodes']
                                if mode == 'train' else config['data.test_episodes'],
                                way_num=config['data.way']
                                if mode == 'train' else config['data.test_way'],
                                image_num=config['data.shot'] + config['data.query']
                                if mode == 'train' else config['data.test_shot'] + config['data.test_query'])
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0, collate_fn=None)

    return dataloader
