from protonets.data.dataloader import get_dataloader


def load(opt, splits):
    ret = {}
    if 'train' in splits:
        ret['train'] = get_dataloader(opt, mode='train')
    if 'val' in splits:
        ret['val'] = get_dataloader(opt, mode='val')
    if 'test' in splits:
        ret['test'] = get_dataloader(opt, mode='test')

    return ret
