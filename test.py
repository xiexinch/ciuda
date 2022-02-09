import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from model.backbone import RepVGG
from model.seg_head import RefineNet
from model.discriminator import FCDiscriminator
from mmseg.ops import resize
from model import least_square_loss
from utils import PolyLrUpdater

if __name__ == '__main__':
    repvgg_a0 = RepVGG(num_blocks=[2, 4, 14, 1],
                       width_multiplier=[0.75, 0.75, 0.75, 2.5],
                       deploy=False,
                       use_se=True,
                       invariant='W')
    repvgg_a0 = repvgg_a0.cuda()
    refinenet = RefineNet(num_classes=19).cuda()
    discriminator = FCDiscriminator(in_channels=19).cuda()
    seg_optimizer = torch.optim.SGD(list(repvgg_a0.parameters()) +
                                    list(refinenet.parameters()),
                                    lr=0.01,
                                    momentum=0.9,
                                    weight_decay=0.0005)

    adv_optimizer = torch.optim.Adam(discriminator.parameters(),
                                     lr=1e-4,
                                     betas=(0.9, 0.999))
    seg_lr_updater = PolyLrUpdater(base_lr=0.01,
                                   max_iters=1000,
                                   min_lr=1e-4,
                                   power=0.9)
    adv_lr_updater = PolyLrUpdater(base_lr=1e-4, max_iters=1000, power=0.9)

    img = Image.open('GOPR0351_frame_000001_rgb_anon.png')
    img = torch.from_numpy(np.array(img).astype(np.float32)).permute(
        2, 0, 1).unsqueeze(0).cuda()

    img = resize(img, scale_factor=0.5)
    feats = repvgg_a0(img)

    seg_out = refinenet(feats)
    print(feats[-1].shape)
    dis_out = discriminator(F.softmax(seg_out))
    print(F.softmax(seg_out))
    print(dis_out.shape)
    target_label = torch.FloatTensor(dis_out.data.size()).fill_(1).cuda()
    source_label = torch.FloatTensor(dis_out.data.size()).fill_(0).cuda()
    print(target_label)
    print(target_label.shape)

    adv_loss = torch.mean((dis_out - target_label).abs()**2)
    print(adv_loss)
    adv_loss = torch.mean((dis_out - source_label).abs()**2)
    print(adv_loss)

    print(least_square_loss(dis_out, source_label))
    ce_loss = torch.nn.CrossEntropyLoss()
    seg_label = torch.randn(seg_out.shape).cuda()
    print(seg_label.shape)
    seg_loss = ce_loss(seg_out, seg_label)
    print(seg_loss)

    # test lr_updater
    # for i in range(1000):
    #     seg_lr_updater.set_lr(seg_optimizer, i)
    #     adv_lr_updater.set_lr(adv_optimizer, i)

    #     print('{0:.8f} {1:.8f}'.format(seg_optimizer.param_groups[0]['lr'],
    #                                    adv_optimizer.param_groups[0]['lr']))
