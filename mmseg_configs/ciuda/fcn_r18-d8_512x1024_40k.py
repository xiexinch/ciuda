_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/rcs_cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(pretrained=None,
             backbone=dict(depth=18,
                           init_cfg=dict(
                               type='Pretrained',
                               checkpoint='open-mmlab://resnet18_v1c')),
             decode_head=dict(
                 in_channels=512,
                 channels=128,
             ),
             auxiliary_head=dict(in_channels=256, channels=64))
