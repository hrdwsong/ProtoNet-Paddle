import os
import json
import math
from tqdm import tqdm

import paddle

from protonets.utils import filter_opt
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
from scripts.averagevaluemeter import AverageValueMeter


def main(opt):
    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    # load model
    model = model_utils.load(opt)
    model_checkpoint = paddle.load(opt['model.model_path'])
    model.set_state_dict(model_checkpoint)
    model.eval()

    # construct data
    data_opt = {'data.' + k: v for k, v in filter_opt(opt, 'data').items()}

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        else:
            data_opt[k] = opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    paddle.seed(1234)

    data = data_utils.load(opt, ['test'])

    meters = {field: AverageValueMeter() for field in opt['log.fields'] }

    model_utils.evaluate(opt, model, data['test'], meters, desc="test")

    for field, meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))
