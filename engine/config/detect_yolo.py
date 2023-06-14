# -*- coding: utf-8 -*-
# @Author  : ouyang
# @Time    : 2023/6/6 9:38

max_epochs = 420
base_lr = 4e-3

# model settings
model = dict(
    backbone=dict(
        type='ResNet50',
        ),
    head=dict(
        type='CHead',
        loss=dict(
            type='MSELoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
    )
)