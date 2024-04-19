_base_ = 'datasets'
# dataset settings
data_type = 'Grounding'
data_root = 'data/charades/'
data = dict(
    train=dict(
        type=data_type,
        label_path=data_root + 'charades_train.jsonl',
        video_path=data_root + 'frames_224_1fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        min_video_len=5,
        fps=1,
        unit=0.1,
        loader=dict(batch_size=32, num_workers=4, pin_memory=True, shuffle=True)),
    val=dict(
        type=data_type,
        label_path=data_root + 'charades_test.jsonl',
        video_path=data_root + 'frames_224_1fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        fps=1,
        unit=0.1,
        loader=dict(batch_size=1, num_workers=4, pin_memory=True, shuffle=False)))
