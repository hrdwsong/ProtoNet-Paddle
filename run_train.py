# import sys
# sys.path.append('/home/aistudio/prototypical-networks-paddle')
import argparse
import paddle
from scripts.train.few_shot.train import main

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
default_dataset = 'miniimagenet'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
parser.add_argument('--data.way', type=int, default=30, metavar='WAY',        # 调试修改，原30
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=1, metavar='SHOT',        # 调试修改，原1
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=15, metavar='QUERY',        # 调试修改，原15
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',        # 调试修改，原5
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=1, metavar='TESTSHOT',        # 调试修改，原1
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',        # 调试修改，原15
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', default=True, help="run in CUDA mode (default: False)")
parser.add_argument('--data_root', type=str, default=r'G:\FSL_filelists\miniImagenet\mini-imagenet-sxc',
                    help="data root directory.")
# parser.add_argument('--data_root', type=str, default=r'/home/aistudio/data/mini-imagenet-sxc',
#                     help="data root directory.")


# model args
default_model_name = 'protonet_conv'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='3,84,84', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")

# train args
parser.add_argument('--train.epochs', type=int, default=400, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
# parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
#                     help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=100, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

args = vars(parser.parse_args())

# 开启0号GPU训练
use_gpu = args['data.cuda']
# use_gpu = False
device = paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

main(args)
