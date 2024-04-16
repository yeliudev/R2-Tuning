_base_ = [
    '../_base_/models/model.py', '../_base_/datasets/charades.py',
    '../_base_/schedules/default.py', '../_base_/runtime.py'
]
# model settings
model = dict(
    adapter_cfg=dict(dropout=0.35), loss_cfg=dict(loss_sal=dict(loss_weight=0.01)))
# runtime settings
stages = dict(
    epochs=50,
    optimizer=dict(lr=2.5e-4),
    lr_schedule=dict(step=[30]),
    validation=dict(nms_cfg=dict(_delete_=True, type='linear')))
