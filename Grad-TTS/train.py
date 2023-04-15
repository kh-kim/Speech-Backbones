# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import json
from datetime import datetime

import numpy as np
from tqdm import tqdm

from scipy.io.wavfile import write

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

# For HiFi-GAN
import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size
use_pre_norm = params.use_pre_norm

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

config = {k:v for k, v in vars(params).items() if not k.startswith('__')}
del config['fix_len_compatibility']


def get_hifigan():
    with open('./checkpts/hifigan-config.json') as f:
        h = AttrDict(json.load(f))
    hifigan = HiFiGAN(h)
    hifigan.load_state_dict(torch.load('./checkpts/hifigan.pt', map_location=lambda loc, storage: loc)['generator'])
    _ = hifigan.cuda().eval()
    hifigan.remove_weight_norm()

    return hifigan


def get_audio(hifigan, y_dec):
    with torch.no_grad():
        audio = hifigan.forward(y_dec).cpu().squeeze().clamp(-1, 1)
    audio = audio.numpy()

    return audio


def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("Config:")
    print(json.dumps(config, indent=4, ensure_ascii=False))

    print('Initializing logger...')
    if params.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(log_dir=log_dir)
    if params.use_wandb:
        import wandb
        wandb.init(project="Grad-TTS", config=config)
        wandb.run.name = params.run_name + '_' + get_now()
        wandb.run.save()

    import os
    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')
    batch_collate = TextMelBatchCollate()

    train_dataset = TextMelDataset(train_filelist_path, cmudict_path, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              collate_fn=batch_collate, drop_last=True,
                              num_workers=4, shuffle=True)
    
    valid_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                              collate_fn=batch_collate, drop_last=True,
                              num_workers=4, shuffle=False)

    print('Initializing model...')
    model = GradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale, use_pre_norm).cuda()
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    if params.use_wandb:
        wandb.watch(model)

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Initializing test batch...')
    test_batch = valid_dataset.sample_test_batch(size=params.test_size)

    to_log = {
        "ground_truth": []
    }
    for i, item in enumerate(test_batch):
        mel = item['y']

        if params.use_tensorboard:
            logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                            global_step=0, dataformats='HWC')
        if params.use_wandb:
            to_log["ground_truth"].append(wandb.Image(
                plot_tensor(mel.squeeze()),
                caption=f'image_{i}',
            ))

        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    if params.use_wandb:
        wandb.log(to_log)

    print('Start training...')
    iteration = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_dur_losses, train_prior_losses, train_diff_losses = [], [], []

        with tqdm(train_loader, total=len(train_dataset) // batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                    if params.use_tensorboard:
                        logger.add_scalar('training/duration_loss', dur_loss.item(),
                                        global_step=iteration)
                        logger.add_scalar('training/prior_loss', prior_loss.item(),
                                        global_step=iteration)
                        logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                        global_step=iteration)
                        logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                        global_step=iteration)
                        logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                        global_step=iteration)
                        logger.add_scalar('training/total_loss', loss.item(),
                                        global_step=iteration)
                    if params.use_wandb:
                        wandb.log({
                            'train_duration_loss': dur_loss.item(),
                            'train_prior_loss': prior_loss.item(),
                            'train_diffusion_loss': diff_loss.item(),
                            'train_encoder_grad_norm': enc_grad_norm,
                            'train_decoder_grad_norm': dec_grad_norm,
                            'train_total_loss': loss.item(),
                        })
                
                train_dur_losses.append(dur_loss.item())
                train_prior_losses.append(prior_loss.item())
                train_diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    msg = "Epoch: %4d, iteration: %8d | dur_loss: %.4f, prior_loss: %.4f, diff_loss: %.4f" % (epoch, iteration, dur_loss.item(), prior_loss.item(), diff_loss.item())
                    progress_bar.set_description(msg)
                
                iteration += 1

        model.eval()
        valid_dur_losses, valid_prior_losses, valid_diff_losses = [], [], []

        with torch.no_grad():
            with tqdm(valid_loader, total=len(valid_dataset) // batch_size) as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                    y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                    dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                         y, y_lengths,
                                                                         out_size=out_size)

                    loss = sum([dur_loss, prior_loss, diff_loss * params.diff_loss_scale])

                    valid_dur_losses.append(dur_loss.item())
                    valid_prior_losses.append(prior_loss.item())
                    valid_diff_losses.append(diff_loss.item())

                    if batch_idx % 5 == 0:
                        msg = f'Epoch: {epoch} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                        msg = "Epoch: %4d | dur_loss: %.4f, prior_loss: %.4f, diff_loss: %.4f" % (epoch, dur_loss.item(), prior_loss.item(), diff_loss.item())
                        progress_bar.set_description(msg)

        log_msg = 'Train Epoch %d: duration loss = %.3f ' % (epoch, np.mean(train_dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(train_prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(train_diff_losses)
        
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        log_msg = 'Valid Epoch %d: duration loss = %.3f ' % (epoch, np.mean(valid_dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(valid_prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(valid_diff_losses)

        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if params.use_tensorboard:
            logger.add_scalar('validation/duration_loss_mean', np.mean(train_dur_losses),
                              global_step=epoch)
            logger.add_scalar('validation/prior_loss_mean', np.mean(train_prior_losses),
                              global_step=epoch)
            logger.add_scalar('validation/diffusion_loss_mean', np.mean(train_diff_losses),
                              global_step=epoch)
            logger.add_scalar('validation/total_loss_mean', np.mean(train_dur_losses) + np.mean(train_prior_losses) + np.mean(train_diff_losses),
                              global_step=epoch)
            logger.add_scalar('validation/duration_loss_median', np.median(train_dur_losses),
                              global_step=epoch)
            logger.add_scalar('validation/prior_loss_median', np.median(train_prior_losses),
                              global_step=epoch)
            logger.add_scalar('validation/diffusion_loss_median', np.median(train_diff_losses),
                              global_step=epoch)
            logger.add_scalar('validation/total_loss_median', np.median(train_dur_losses) + np.median(train_prior_losses) + np.median(train_diff_losses),
                              global_step=epoch)
        if params.use_wandb:
            wandb.log({
                'valid_duration_loss_mean': np.mean(train_dur_losses),
                'valid_prior_loss_mean': np.mean(train_prior_losses),
                'valid_diffusion_loss_mean': np.mean(train_diff_losses),
                'valid_total_loss_mean': np.mean(train_dur_losses) + np.mean(train_prior_losses) + np.mean(train_diff_losses),
                'valid_duration_loss_median': np.median(train_dur_losses),
                'valid_prior_loss_median': np.median(train_prior_losses),
                'valid_diffusion_loss_median': np.median(train_diff_losses),
                'valid_total_loss_median': np.median(train_dur_losses) + np.median(train_prior_losses) + np.median(train_diff_losses),
            })

        if epoch % params.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            hifigan = get_hifigan()

            to_log = {
                "generated_enc": [],
                "generated_dec": [],
                "alignment": [],
                "generated_audio": [],
            }

            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=100, temperature=20.0)

                if params.use_tensorboard:
                    logger.add_image(f'image_{i}/generated_enc',
                                     plot_tensor(y_enc.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/generated_dec',
                                     plot_tensor(y_dec.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/alignment',
                                     plot_tensor(attn.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                if params.use_wandb:
                    to_log['generated_enc'].append(wandb.Image(plot_tensor(y_enc.squeeze().cpu()), caption=f'image_{i}'))
                    to_log['generated_dec'].append(wandb.Image(plot_tensor(y_dec.squeeze().cpu()), caption=f'image_{i}'))
                    to_log['alignment'].append(wandb.Image(plot_tensor(attn.squeeze().cpu()), caption=f'image_{i}'))

                    to_log['generated_audio'].append(
                        wandb.Audio(get_audio(hifigan, y_dec), sample_rate=params.sample_rate, caption=f'audio_{i}')
                    )

                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_dir}/alignment_{i}.png')
                
            if params.use_wandb:
                wandb.log(to_log)

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
