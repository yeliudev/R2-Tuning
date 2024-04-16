_base_ = [
    '../_base_/models/model.py', '../_base_/datasets/tacos.py',
    '../_base_/schedules/default.py', '../_base_/runtime.py'
]
# model settings
model = dict(loss_cfg=dict(loss_sal=dict(loss_weight=0.05)))
# runtime settings
stages = dict(
    epochs=100,
    optimizer=dict(lr=2.5e-4),
    lr_schedule=dict(step=[50]),
    validation=dict(nms_cfg=dict(_delete_=True, type='linear')))
