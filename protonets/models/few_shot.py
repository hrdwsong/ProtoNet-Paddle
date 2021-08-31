import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import create_parameter

from protonets.models import register_model

from .utils import euclidean_dist


class Flatten(nn.Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape([x.shape[0], -1])


class Protonet(nn.Layer):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder
        self.learnable_scale = create_parameter(shape=[1], dtype='float32',
                                                default_initializer=nn.initializer.Constant(1.0))

    def loss(self, sample):
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = xs.shape[0]
        assert xq.shape[0] == n_class
        n_support = xs.shape[1]
        n_query = xq.shape[1]

        target_inds = paddle.arange(0, n_class).reshape([n_class, 1, 1]).expand([n_class, n_query, 1])
        target_inds.stop_gradient = True

        x = paddle.concat([xs.reshape([n_class * n_support, *xs.shape[2:]]),
                           xq.reshape([n_class * n_query, *xq.shape[2:]])], 0)

        z = self.encoder.forward(x)
        z_dim = z.shape[-1]

        z_proto = z[:n_class * n_support].reshape([n_class, n_support, z_dim]).mean(1)
        zq = z[n_class * n_support:]

        dists = self.learnable_scale * euclidean_dist(zq, z_proto) / 1600

        # loss_val = F.cross_entropy(-dists, target_inds)
        # y_hat = dists.argmin(axis=1)

        log_p_y = F.log_softmax(-dists, axis=1).reshape([n_class, n_query, -1])
        # loss_val = -log_p_y.gather(axis=2, index=paddle.arange(0, n_class)).squeeze().reshape(-1).mean()
        loss_val = []  # 以下是pytorch gather函数的等效实现
        for i in range(log_p_y.shape[0]):
            loss_val.append(-log_p_y[i, :, i])
        loss_val = paddle.concat(loss_val, axis=0)
        loss_val = loss_val.mean()
        y_hat = log_p_y.argmax(axis=2)

        acc_val = paddle.equal(y_hat, target_inds.squeeze()).astype('float32').mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 3, padding=1,
                      weight_attr=nn.initializer.KaimingUniform(),
                      bias_attr=nn.initializer.KaimingUniform()),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.MaxPool2D(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)

