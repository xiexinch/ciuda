_base_ = [
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='RepVGG',
                  num_blocks=[2, 4, 14, 1],
                  width_multiplier=[0.75, 0.75, 0.75, 2.5],
                  deploy=False,
                  use_se=True,
                  invariant='W'),
    decode_head=dict(type='RefineNet',
                     in_channels=[1280, 192, 96, 48],
                     channels=256,
                     num_classes=19,
                     in_index=[0, 1, 2, 3],
                     input_transform='multiple_select'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
