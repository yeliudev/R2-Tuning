_base_ = 'datasets'
# dataset settings
data_type = 'Grounding'
data_root = 'data/qvhighlights/'
data = dict(
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=data_type,
            label_path=data_root + 'qvhighlights_train.jsonl',
            video_path=data_root + 'frames_224_0.5fps',
            cache_path=data_root + 'clip_b32_vid_k4',
            query_path=data_root + 'clip_b32_txt_k4',
            use_cache=True,
            min_video_len=5,
            fps=0.5,
            unit=2),
        loader=dict(batch_size=128, num_workers=4, pin_memory=True, shuffle=True)),
    val=dict(
        type=data_type,
        label_path=data_root + 'qvhighlights_val.jsonl',
        video_path=data_root + 'frames_224_0.5fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        fps=0.5,
        unit=2,
        loader=dict(batch_size=1, num_workers=4, pin_memory=True, shuffle=False)),
    test=dict(
        type=data_type,
        label_path=data_root + 'qvhighlights_test.jsonl',
        video_path=data_root + 'frames_224_0.5fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        fps=0.5,
        unit=2,
        loader=dict(batch_size=1, num_workers=4, pin_memory=True, shuffle=False)))
