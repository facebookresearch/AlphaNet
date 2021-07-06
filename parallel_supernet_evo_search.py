# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import models
from utils.config import setup
import utils.comm as comm
import utils.saver as saver

from data.data_loader import build_data_loader
from evaluate import attentive_nas_eval as attentive_nas_eval
import utils.logging as logging
import argparse

"""
    using multiple nodes to run evolutionary search:
    1) each GPU will evaluate its own sub-networks
    2) all evaluation results will be aggregated on GPU 0
"""
parser = argparse.ArgumentParser(description='Test AlphaNet Models')
parser.add_argument('--config-file', default='./configs/parallel_supernet_evo_search.yml')
parser.add_argument('--machine-rank', default=0, type=int, 
                    help='machine rank, distributed setting')
parser.add_argument('--num-machines', default=1, type=int, 
                    help='number of nodes, distributed setting')
parser.add_argument('--dist-url', default="tcp://127.0.0.1:10001", type=str, 
                    help='init method, distributed setting')
parser.add_argument('--seed', default=1, type=int, 
                    help='default random seed')
run_args = parser.parse_args()


logger = logging.get_logger(__name__)


def eval_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # local rank, local machine cuda id
    args.local_rank = args.gpu
    args.batch_size = args.batch_size_per_gpu

    global_rank = args.gpu + args.machine_rank * ngpus_per_node
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=global_rank
    )

    # Setup logging format.
    logging.setup_logging("stdout.log", 'w')

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    args.rank = comm.get_rank() # global rank
    torch.cuda.set_device(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # build the supernet
    logger.info("=> creating model '{}'".format(args.arch))
    model = models.model_factory.create_model(args)
    model.cuda(args.gpu)
    model = comm.get_parallel_model(model, args.gpu) #local rank

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    ## load dataset, train_sampler: distributed
    train_loader, val_loader, train_sampler =  build_data_loader(args)

    assert args.resume
    #reloading model
    model.module.load_weights_from_pretrained_models(args.resume)

    if train_sampler:
        train_sampler.set_epoch(0)

    targeted_min_flops = args.evo_search.targeted_min_flops
    targeted_max_flops = args.evo_search.targeted_max_flops

    # run evolutionary search
    parent_popu = []
    for idx in range(args.evo_search.parent_popu_size):
        if idx == 0:
            cfg = model.module.sample_min_subnet()
        else:
            cfg = model.module.sample_active_subnet_within_range(
                targeted_min_flops, targeted_max_flops
            )
        cfg['net_id'] = f'net_{idx % args.world_size}_evo_0_{idx}'
        parent_popu.append(cfg)

    pareto_global = {}
    for evo in range(args.evo_search.evo_iter):   
        # partition the set of candidate sub-networks
        # and send them to each GPU for parallel evaluation

        # sub-networks to be evaluated on GPU {args.rank}
        my_subnets_to_be_evaluated = {}
        n_evaluated = len(parent_popu) // args.world_size * args.world_size
        for cfg in parent_popu[:n_evaluated]:
            if cfg['net_id'].startswith(f'net_{args.rank}_'):
                my_subnets_to_be_evaluated[cfg['net_id']] = cfg

        # aggregating all evaluation results
        eval_results = attentive_nas_eval.validate(
            my_subnets_to_be_evaluated,
            train_loader,
            val_loader,
            model,
            criterion, 
            args, 
            logger,
        )

        # update the Pareto frontier
        # in this case, we search the best FLOPs vs. accuracy trade-offs
        for cfg in eval_results:
            f = round(cfg['flops'] / args.evo_search.step) * args.evo_search.step
            if f not in pareto_global or pareto_global[f]['acc1'] < cfg['acc1']:
                pareto_global[f] = cfg

        # next batch of sub-networks to be evaluated
        parent_popu = []
        # mutate 
        for idx in range(args.evo_search.mutate_size):
            while True:
                old_cfg = random.choice(list(pareto_global.values()))
                cfg = model.module.mutate_and_reset(old_cfg, prob=args.evo_search.mutate_prob)
                flops = model.module.compute_active_subnet_flops()
                if flops >= targeted_min_flops and flops <= targeted_max_flops:
                    break
            cfg['net_id'] = f'net_{idx % args.world_size}_evo_{evo}_mutate_{idx}'
            parent_popu.append(cfg)

        # cross over
        for idx in range(args.evo_search.crossover_size):
            while True:
                cfg1 = random.choice(list(pareto_global.values()))
                cfg2 = random.choice(list(pareto_global.values()))
                cfg = model.module.crossover_and_reset(cfg1, cfg2)
                flops = model.module.compute_active_subnet_flops()
                if flops >= targeted_min_flops and flops <= targeted_max_flops:
                    break
            cfg['net_id'] = f'net_{idx % args.world_size}_evo_{evo}_crossover_{idx}'
            parent_popu.append(cfg)

if __name__ == '__main__':
    # setup enviroments
    args = setup(run_args.config_file)
    args.dist_url = run_args.dist_url
    args.machine_rank = run_args.machine_rank
    args.num_nodes = run_args.num_machines

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.num_nodes
        assert args.world_size > 1, "only support DDP settings"
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # eval_worker process function
        mp.spawn(eval_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError
 
