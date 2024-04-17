# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import argparse
from datetime import timedelta

import nncore
from nncore.engine import Engine, comm, set_random_seed
from nncore.nn import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--checkpoint', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--work_dir', help='working directory')
    parser.add_argument('--eval', help='evaluation mode', action='store_true')
    parser.add_argument('--dump', help='dump inference outputs', action='store_true')
    parser.add_argument('--seed', help='random seed', type=int)
    parser.add_argument('--amp', help='amp data type', type=str, default='fp16')
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist(launcher=args.launcher, timeout=timedelta(hours=1))

    if comm.is_main_process() and not args.eval and not args.dump:
        if args.work_dir is None:
            work_dir = nncore.join('work_dirs', nncore.pure_name(args.config))
            work_dir = nncore.mkdir(work_dir, modify_path=True)
        else:
            work_dir = args.work_dir

        time_stp = nncore.get_timestamp()
        log_file = nncore.join(work_dir, '{}.log'.format(time_stp))
    else:
        log_file = work_dir = None

    logger = nncore.get_logger(log_file=log_file)
    logger.info(f'Environment info:\n{nncore.collect_env_info()}')
    logger.info(f'Launcher: {launcher}')
    logger.info(f'Config: {cfg.text}')

    seed = args.seed if args.seed is not None else cfg.get('seed')
    seed = set_random_seed(seed, deterministic=True)
    logger.info(f'Using random seed: {seed}')

    model = build_model(cfg.model, dist=bool(launcher))
    logger.info(f'Model architecture:\n{model.module}')

    params = []
    logger.info('Learnable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'{name} - {param.shape}')
            params.append(param)

    train_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = round(train_params / total_params * 100, 3)
    param = round(train_params / 1024 / 1024, 3)
    logger.info(f'Learnable Parameters: {param}M ({ratio}%)')

    if not args.eval and not args.dump:
        world_size = comm.get_world_size()
        per_gpu_bs = int(cfg.data.train.loader.batch_size / world_size)
        cfg.data.train.loader.batch_size = per_gpu_bs
        logger.info(f'Auto Scale Batch Size: {world_size} GPU(s) * {per_gpu_bs} Samples')

    if args.dump:
        cfg.stages.validation.dump_template = 'hl_{}_submission.jsonl'
        args.eval = 'both'

    engine = Engine(
        model,
        cfg.data,
        stages=cfg.stages,
        hooks=cfg.hooks,
        work_dir=work_dir,
        seed=seed,
        amp=args.amp,
        debug=args.debug)

    if checkpoint := args.checkpoint:
        engine.load_checkpoint(checkpoint)
    elif checkpoint := args.resume:
        engine.resume(checkpoint)

    engine.launch(eval=args.eval)


if __name__ == '__main__':
    main()
