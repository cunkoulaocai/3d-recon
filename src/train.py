import os

from tqdm import tqdm

import metrics
from configs.config import Config

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = Config()

import torch as tr

if config.use_gpu:
    print('mode: GPU')
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

from gan import GAN
from data_loaders import data_loader
from tensorboardX.writer import SummaryWriter

file_path = os.path.dirname(os.path.realpath('.'))
dir_path = os.path.dirname(os.path.join(file_path, 'data'))

config = Config()

args_data = 'chairs_pose0.5'

data_setting = config.category_settings[args_data]
data_train = data_setting['train']
data_val = data_setting['val']

train_setting = []
for cate, split, p_ratio, nop_ratio in data_train:
    p = os.path.join(dir_path, "data", "%s_32x32alpha_perspective_d100_r32_vp24_random_default" % cate, split)
    train_setting.append((p, p_ratio, nop_ratio))

val_setting = []
for cate, split in data_val:
    p = os.path.join(dir_path, "data", "%s_32x32alpha_perspective_d100_r32_vp24_random_default" % cate, split)
    val_setting.append(p)

# Training DataLoader
train_loader = data_loader.TrainDataLoader(
    train_setting,
    shape=[config.resolution, config.resolution, 1],
    viewpoints=config.viewpoints,
    binarize_data=config.binarize_data,
    batch_size=config.batch_size,
    shuffle=True
)

# Validation DataLoader
val_loader = data_loader.ValDataLoader(
    val_setting,
    shape=[config.resolution, config.resolution, 1],
    viewpoints=config.viewpoints,
    binarize_data=config.binarize_data
)

# Noise Samplers for embeddings Z and pose P
z_sampler = data_loader.ZSampler(config.use_normal_noise)
p_sampler = data_loader.PSampler()

# Gan Model
gan = GAN(config)


def validate(batch, iter_no):
    # x1 = tr.tensor(batch['img_1']).permute([0, 3, 1, 2])
    # x2 = tr.tensor(batch['img_2']).permute([0, 3, 1, 2])
    # p1 = tr.tensor(batch['pos_1'])
    # p2 = tr.tensor(batch['pos_2'])

    x = tr.tensor(batch['image']).permute([0, 3, 1, 2]).cuda()
    gt_vox = tr.tensor(batch['vox']).cuda()
    p = tr.tensor(batch['pose']).cuda()

    pred_vox = gan.predict_voxel(x)

    t05_iou = metrics.iou_t(gt_vox, pred_vox, threshold=0.5).mean()
    t04_iou = metrics.iou_t(gt_vox, pred_vox, threshold=0.4).mean()
    max_iou = metrics.maxIoU(gt_vox, pred_vox)
    avg_precision = metrics.average_precision(gt_vox, pred_vox)

    val_writer.add_scalar('t05_iou', t05_iou, iter_no)
    val_writer.add_scalar('t04_iou', t04_iou, iter_no)
    val_writer.add_scalar('max_iou', max_iou, iter_no)
    val_writer.add_scalar('avg_precision', avg_precision, iter_no)


def train_step(iter_no, batch):
    global g_iter_count, d_iter_count, mode

    x1 = tr.tensor(batch['img_1']).permute([0, 3, 1, 2]).cuda()
    x2 = tr.tensor(batch['img_2']).permute([0, 3, 1, 2]).cuda()
    p1 = tr.tensor(batch['pos_1']).cuda()
    p2 = tr.tensor(batch['pos_2']).cuda()

    x = tr.tensor(batch['img']).permute([0, 3, 1, 2]).cuda()
    z = tr.tensor(z_sampler(config.batch_size, config.z_dim))
    p = tr.tensor(p_sampler(config.batch_size))

    l_ae, l_recon, l_pinv, l_vinv = gan.step_train_autoencoder(x1, p1, x2, p2)
    d_loss, g_loss, d_acc, g_acc = gan.step_train_gan(x, z, p, mode)

    if iter_no % config.logging_interval == 0:
        train_writer.add_scalar('loss_ae', l_ae, iter_no)
        train_writer.add_scalar('loss_recon', l_recon, iter_no)
        train_writer.add_scalar('loss_pinv', l_pinv, iter_no)
        train_writer.add_scalar('loss_vinv', l_vinv, iter_no)
        train_writer.add_scalar('loss_disc', d_loss, iter_no)
        train_writer.add_scalar('loss_gen', g_loss, iter_no)
        train_writer.add_scalar('acc_disc', d_acc, iter_no)
        train_writer.add_scalar('acc_gen', g_acc, iter_no)

    if mode == 'G':
        g_iter_count += 1
        if g_iter_count == config.max_g_iters:
            g_iter_count = 0
            mode = 'D'
    else:
        d_iter_count += 1
        if d_iter_count == config.max_d_iters:
            d_iter_count = 0
            mode = 'G'


iter_no = 0
max_iters = 100

mode = 'G'
g_iter_count = 0
d_iter_count = 0

batch_size = config.batch_size

train_writer = SummaryWriter(log_dir='../logs/train')
val_writer = SummaryWriter(log_dir='../logs/val')

with tqdm(total=max_iters) as pbar:
    for iter_no in range(max_iters):
        train_batch = train_loader.next_batch()

        gan.train()
        train_step(iter_no, train_batch)

        if iter_no % config.validation_interval == 0:
            val_batch = val_loader.next_batch()
            gan.eval()
            validate(val_batch, iter_no)
            pbar.update(1)
