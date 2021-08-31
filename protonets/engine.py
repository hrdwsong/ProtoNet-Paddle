from tqdm import tqdm
import paddle


class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = {}
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'opt': kwargs['opt'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0,  # epochs done so far
            't': 0,  # samples seen so far
            'batch': 0,  # samples seen in current epoch
            'stop': False
        }
        state['scheduler'] = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=state['optim_config']['learning_rate'], patience=10,
                                                                 factor=0.3, verbose=True)
        # state['scheduler'] = paddle.optimizer.lr.StepDecay(learning_rate=state['optim_config']['learning_rate'],
        #                                                    step_size=20,
        #                                                    gamma=0.5,
        #                                                    verbose=True)
        state['optim_config']['learning_rate'] = state['scheduler']
        state['optimizer'] = state['optim_method'](parameters=state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                if state['opt']['data.dataset'] == 'omniglot':
                    state['sample'] = sample
                else:
                    n_shot = state['opt']['data.shot']
                    n_query = state['opt']['data.query']
                    n_way = state['opt']['data.way']
                    imgs = sample[0]
                    labels = sample[1]
                    state['sample'] = {}
                    imgs = imgs.reshape([n_way, n_shot+n_query, *imgs.shape[1:]])
                    state['sample']['xs'] = imgs[:, :n_shot, :, :, :]
                    state['sample']['xq'] = imgs[:, n_shot:, :, :, :]

                self.hooks['on_sample'](state)

                state['optimizer'].clear_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
