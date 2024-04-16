_base_ = [
    '../_base_/models/model.py', '../_base_/datasets/ego4d.py',
    '../_base_/schedules/default.py', '../_base_/runtime.py'
]
# model settings
model = dict(adapter_cfg=dict(dropout=0.3))
# runtime settings
stages = dict(
    optimizer=dict(lr=2.5e-4),
    validation=dict(nms_cfg=dict(_delete_=True, type='linear')))
