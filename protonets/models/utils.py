import paddle


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    assert d == y.shape[1]

    x = x.unsqueeze(1).expand([n, m, d])
    y = y.unsqueeze(0).expand([n, m, d])

    return paddle.pow(x - y, 2).sum(2)
