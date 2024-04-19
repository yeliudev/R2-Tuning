_base_ = 'datasets'
# dataset settings
data_type = 'TVSum'
data_root = 'data/tvsum/'
data = dict(
    train=dict(
        type=data_type,
        label_path=data_root + 'tvsum_anno.json',
        video_path=data_root + 'frames_224_0.5fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        loader=dict(batch_size=4, num_workers=4, pin_memory=True, shuffle=True)),
    val=dict(
        type=data_type,
        label_path=data_root + 'tvsum_anno.json',
        video_path=data_root + 'frames_224_0.5fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        loader=dict(batch_size=1, num_workers=4, pin_memory=True, shuffle=False)))
