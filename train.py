import adabound
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import yaml

from addict import Dict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from model.danet import DANet
from model.canet import CANet
from model.drn import drn_d_22, drn_d_38
from utils.checkpoint import save_checkpoint, resume
from utils.class_weight import get_class_weight
from utils.dataset import PASCALVOC
from utils.mean import get_mean, get_std
from utils.transforms import ToTensor, Resize, RandomCrop, RandomFlip, Normalize


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='semantic segmentation using PASCAL VOC')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--resume', action='store_true',
                        help='if you start training from checkpoint')

    return parser.parse_args()


def one_hot(label, n_classes, dtype, device, requires_grad=False):
    # shape => (N, n_classes)
    one_hot_label = torch.eye(
        n_classes, dtype=dtype, requires_grad=requires_grad, device=device)[label]
    return one_hot_label


def train(model, train_loader, criterion, optimizer, config, device):
    model.train()

    epoch_loss = 0.0
    for sample in tqdm.tqdm(train_loader, total=len(train_loader)):
        x = sample['image']
        t = sample['label']
        x = x.to(device)
        t = t.to(device)
        _, _, H, W = x.shape

        h = model(x)
        if config.attention == 'dual':
            loss = 0.0
            for weighted_h in h:
                weighted_h = F.interpolate(
                    weighted_h, (H, W), mode='bilinear', align_corners=False)
                loss += criterion(weighted_h, t)
        else:
            h = F.interpolate(h, (H, W), mode='bilinear')
            loss = criterion(h, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validation(model, test_loader, criterion, config, device):
    model.eval()

    intersections = torch.zeros(config.n_classes).to(device)
    unions = torch.zeros(config.n_classes).to(device)

    eval_loss = 0.0

    for sample in test_loader:
        x, t = sample['image'], sample['label']

        x = x.to(device)
        t = t.to(device)
        _, _, H, W = x.shape

        with torch.no_grad():
            h = model(x)

            if config.attention == 'dual':
                loss = 0.0
                for weighted_h in h:
                    weighted_h = F.interpolate(
                        weighted_h, (H, W), mode='bilinear', align_corners=False)
                    loss += criterion(weighted_h, t)
                # using the only output from dual attention modules
                h = F.interpolate(
                    h[0], (H, W), mode='bilinear', align_corners=False)
            else:
                h = F.interpolate(
                    h, (H, W), mode='bilinear', align_corners=False)
                loss = criterion(h, t)

            eval_loss += loss

            _, pred = h.max(1)    # y_pred.shape => (N, H, W)

            # ignore void class(255)
            mask = (t >= 0) & (t < config.n_classes)
            t = t[mask]
            pred = pred[mask]

            pred = one_hot(pred, config.n_classes, torch.long, device)
            t = one_hot(t, config.n_classes, torch.long, device)

            intersection = torch.sum(pred & t, dim=0)
            union = torch.sum(pred | t, dim=0)

            intersections += intersection.float()
            unions += union.float()

    """ iou[i] is the IoU of class i """
    iou = intersections / unions
    eval_loss = eval_loss / len(test_loader)

    return iou, eval_loss.item()


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    # DataLoaders
    train_data = PASCALVOC(
        CONFIG,
        mode="train",
        transform=Compose([
            RandomCrop(CONFIG),
            Resize(CONFIG),
            RandomFlip(),
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ])
    )

    val_data = PASCALVOC(
        CONFIG,
        mode="val",
        transform=Compose([
            RandomCrop(CONFIG),
            Resize(CONFIG),
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ])
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if CONFIG.attention == 'dual':
        model = DANet(CONFIG)
        print('Dual Attintion modules will be added to this base model')
    elif CONFIG.attention == 'channel':
        model = CANet(CONFIG)
        print('Channel Attintion modules will be added to this base model')
    else:
        if CONFIG.model == 'drn_d_22':
            print(
                'Dilated ResNet D 22 w/o Dual Attention modules will be used as a model.')
            model = drn_d_22(pretrained=True, num_classes=CONFIG.n_classes)
        elif CONFIG.model == 'drn_d_38':
            print(
                'Dilated ResNet D 28 w/o Dual Attention modules will be used as a model.')
            model = drn_d_38(pretrained=True, num_classes=CONFIG.n_classes)
        else:
            print('There is no option you chose as a model.')
            print(
                'Therefore, Dilated ResNet D 22 w/o Dual Attention modules will be used as a model.')
            model = drn_d_22(pretrained=True, num_classes=CONFIG.n_classes)

    # set optimizer, lr_scheduler
    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov)
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            model.parameters(),
            lr=CONFIG.learning_rate,
            final_lr=CONFIG.final_lr,
            weight_decay=CONFIG.weight_decay)
    else:
        print('There is no optimizer which suits to your option. \
            Instead, SGD will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov)

    # learning rate scheduler
    if CONFIG.optimizer == 'SGD':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience)
    else:
        scheduler = None

    # send the model to cuda/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.benchmark = True

    # resume if you want
    begin_epoch = 0
    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            begin_epoch, model, optimizer, scheduler = \
                resume(CONFIG, model, optimizer, scheduler)
            print('training will start from {} epoch'.format(begin_epoch))

    # criterion for loss
    if CONFIG.class_weight:
        criterion = nn.CrossEntropyLoss(
            weight=get_class_weight().to(device),
            ignore_index=255
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    # train and validate model
    print('\n------------------------Start training------------------------\n')
    losses_train = []
    losses_val = []
    val_ious = []
    mean_ious = []
    mean_ious_without_bg = []
    best_mean_iou = 0.0

    for epoch in range(begin_epoch, CONFIG.max_epoch):
        # training
        loss_train = train(
            model, train_loader, criterion, optimizer, CONFIG, device)
        losses_train.append(loss_train)

        # validation
        val_iou, loss_val = validation(
            model, val_loader, criterion, CONFIG, device)
        val_ious.append(val_iou)
        losses_val.append(loss_val)
        if CONFIG.optimizer == 'SGD':
            scheduler.step(loss_val)

        mean_ious.append(val_ious[-1].mean().item())
        mean_ious_without_bg.append(val_ious[-1][1:].mean().item())

        # save checkpoint every 5 epoch
        if epoch % 5 == 0 and epoch != 0:
            save_checkpoint(CONFIG, epoch, model, optimizer, scheduler)

        # save a model every 50 epoch
        if epoch % 50 == 0 and epoch != 0:
            torch.save(
                model.state_dict(), os.path.join(CONFIG.result_path, 'epoch_{}_model.prm'.format(epoch)))

        if best_mean_iou < mean_ious[-1]:
            best_mean_iou = mean_ious[-1]
            torch.save(
                model.state_dict(), os.path.join(CONFIG.result_path, 'best_mean_iou_model.prm'))

        # tensorboardx
        if writer:
            writer.add_scalars(
                "loss", {
                    'loss_train': losses_train[-1],
                    'loss_val': losses_val[-1]}, epoch)
            writer.add_scalar(
                "mean_iou", mean_ious[-1], epoch)
            writer.add_scalar(
                "mean_iou_w/o_bg", mean_ious_without_bg[-1], epoch)

        print(
            'epoch: {}\tloss_train: {:.5f}\tloss_val: {:.5f}\tmean IOU: {:.3f}\tmean IOU w/o bg: {:.3f}'.format(
                epoch, losses_train[-1], losses_val[-1], mean_ious[-1], mean_ious_without_bg[-1])
        )

    torch.save(
        model.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))


if __name__ == '__main__':
    main()
