from tqdm import tqdm
import paddle

from protonets.utils import filter_opt
from protonets.models import get_model


def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)


def evaluate(opt, model, data_loader, meters, desc=None):
    model.eval()

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    if opt['data.dataset'] == 'miniimagenet':
        with paddle.no_grad():
            for sample in data_loader:
                n_shot = opt['data.test_shot']
                n_query = opt['data.test_query']
                n_way = opt['data.test_way']
                imgs = sample[0]
                labels = sample[1]
                imgs = imgs.reshape([n_way, n_shot + n_query, *imgs.shape[1:]])
                xs = imgs[:, :n_shot, :, :, :]
                xq = imgs[:, n_shot:, :, :, :]
                sample_test = {'xs': xs, 'xq': xq}
                _, output = model.loss(sample_test)
                for field, meter in meters.items():
                    meter.add(output[field])

    return meters
