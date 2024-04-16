# runtime settings
stages = dict(
    epochs=30,
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=1e-4),
    lr_schedule=dict(type='epoch', policy='step', step=[20]),
    warmup=dict(type='iter', policy='linear', steps=500, ratio=0.001),
    grad_clip=dict(max_norm=35, norm_type=2),
    validation=dict(interval=1))
