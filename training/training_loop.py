﻿# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
from pathlib import Path

import psutil
import PIL.Image
import numpy as np
import torch
from cv2 import cv2

import dnnlib
from models.yolo import Model
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

#----------------------------------------------------------------------------
from utils.general import labels_to_class_weights, intersect_dicts, xywh2xyxy
from utils.torch_utils import de_parallel

ROOT = Path(__file__).resolve().parents[0]

def label_preparetion(labels, nums = 256):
    new_bbox, new_label = [], []
    for batch, label in enumerate(labels):
        if label.shape[0] < nums:
            tmp = torch.zeros([nums-label.shape[0],6])
            label[:, 1] = 1
            label = torch.cat([label, tmp], dim=0)

        new_bbox.append(label[:nums, 1:])

    return np.stack(new_bbox)


def generate_labels(batch_size=8, seed=0, nums = 256):
    new_labels = []
    np.random.seed(seed)
    for idx in range(batch_size):
        rnd = np.random.randint(20, 40)
        rnd_level = np.random.choice(np.linspace(1, 4, 31), 1)
        batch = int(800 / (rnd_level * rnd))
        init_ = np.random.choice(np.linspace(20, 200, 181), batch * 2).reshape(batch, 2) / 255
        shape_ = (np.random.choice(np.linspace(-3, 3, 7), batch * 2).reshape(batch, 2) + rnd) / 255
        head_ = np.zeros([batch, 2])
        head_[:, 0] = idx
        new_labels.append(torch.from_numpy(np.concatenate([head_, init_, shape_], -1)))

    return new_labels


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.img_size, 7, 32)
    gh = np.clip(4320 // training_set.img_size, 4, 32)

    all_indices = list(range(len(training_set)))

    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    old_images, images, labels, paths, shapes = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(old_images), np.stack(images), labels, np.stack(paths), np.stack(shapes)


#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def plot(label, batch, run_dir):
    print("generate groundTrue bbox...")
    gt_img = np.ones([batch, 256, 258, 3]) * 255
    gt_img[:, :, 257:, ] = 0
    for idx, bboxs in enumerate(label):
        for bbox in bboxs:
            (x, y, w, h) = bbox[2:] * 255
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            cv2.rectangle(gt_img[idx], (x1,y1),(x2,y2), (255, 0, 255), 2)

    gt_img = np.concatenate(gt_img, 1)
    cv2.imwrite(run_dir, gt_img)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    # data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    Y_kwargs                = {},       # Options for Yolo network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    **others,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    # device = torch.device('cuda', 1)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    hyp, nc, ycfg, ch, imgsz, label_smoothing, names = others['hyp'], others['nc'], others['ycfg'], others['ch'], others['imgsz'], others['label_smoothing'], others['names']
    # Load training set.
    if rank == 0:
        print('Loading training set...')

    training_loader, training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_iterator = iter(training_loader)
    # training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    # training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    # training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', others['imgsz'])

        print()


    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(img_resolution=imgsz, img_channels=ch)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Y = dnnlib.util.construct_class_by_name(**Y_kwargs).eval().to(device)  # subclass of torch.nn.Module

    ckpt = torch.load(ROOT / 'best.pt')
    Y = Model(ycfg or ckpt['model'].yaml, ch=ch, nc=nc, anchors=hyp.get('anchors')).eval().to(device)
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, Y.state_dict(), exclude=[])  # intersect
    Y.load_state_dict(csd, strict=False)

    nc = others['nc']
    nl = de_parallel(Y).model[-1].nl
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = label_smoothing
    Y.nc = nc  # attach number of classes to model
    Y.hyp = hyp  # attach hyperparameters to model
    Y.class_weights = labels_to_class_weights(training_set.labels, nc).to(device) * nc  # attach class weights
    Y.names = names


    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)], :
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    _label = generate_labels(batch_gpu, seed=int(time.time() % 20))

    plot(_label, batch_gpu, os.path.join(run_dir, 'gt_label_init.png'))

    gen_yolo_label = torch.cat(_label).to(device)
    gen_label = torch.from_numpy(label_preparetion(_label)).to(device)
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        img = misc.print_module_summary(G, [z, gen_label])
        misc.print_module_summary(D, [img])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_conditions', G.conditions),  ('G_synthesis', G.synthesis), ('D', D), ('Y', Y), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_labels = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, old_images, aug_images, labels, paths, shapes = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(old_images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        save_image_grid(aug_images, os.path.join(run_dir, 'reals_aug.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([old_images.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_labels = torch.from_numpy(label_preparetion(labels)).to(device)
        grid_labels = grid_labels.split(batch_gpu)
        images = torch.cat([G_ema(z=z, bbox=label, noise_mode='const').cpu() for z, label in zip(grid_z, grid_labels)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_oimg, phase_real_img,  phase_real_olabel, phase_real_label, _, _ = next(training_set_iterator)

            if ch == 1:
                phase_real_oimg = phase_real_oimg[:,0].unsqueeze(1)
                phase_real_img = phase_real_img[:,0].unsqueeze(1)

            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_oimg = (phase_real_oimg.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

            phase_real_olabel = phase_real_olabel.to(device)

            phase_real_label = torch.from_numpy(label_preparetion(phase_real_label)).to(device).to(torch.float32)
            phase_real_label = phase_real_label.split(batch_gpu)

            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]


        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img,  real_label,  gen_z) in enumerate(zip(phase_real_oimg, phase_real_label, phase_gen_z)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_label=real_label, real_olabel=phase_real_olabel, gen_z=gen_z, gen_label=gen_label, gen_olabel=gen_yolo_label, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, bbox=label, noise_mode='const').cpu() for z, label in zip(grid_z, grid_labels)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            # for metric in metrics:
            #     result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
            #         dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
            #     if rank == 0:
            #         metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
            #     stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
