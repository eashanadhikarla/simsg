'''
!/usr/bin/python

Copyright 2018 Google LLC
Modification copyright 2020 Helisa Dhamo, Azade Farshad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

--------------------------------------------------------------------------

Authors       : Eashan Adhikarla & Jitong Ding
Date Created  : Apr 09' 2021
Data Modified : Apr 29' 2021
Description   : Script to train SIMSG | CRN - 64 & SPADE - 64

'''

import argparse

import os
import math
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from simsg.data import imagenet_deprocess_batch

from simsg.discriminators import PatchDiscriminator, AcCropDiscriminator, MultiscaleDiscriminator, divide_pred
from simsg.losses import get_gan_losses, gan_percept_loss, GANLoss, VGGLoss
from simsg.metrics import jaccard
from simsg.model import SIMSGModel
from simsg.utils import int_tuple
from simsg.utils import timeit, bool_flag, LossManager

from simsg.loader_utils import build_train_loaders
from scripts.utils import *

torch.backends.cudnn.benchmark = True


def build_model(config, vocab):
  if config['checkpoint_start_from'] is not None:
    checkpoint = torch.load(config['checkpoint_start_from'])
    kwargs = checkpoint['model_kwargs']
    model = SIMSGModel(**kwargs)
    raw_state_dict = checkpoint['model_state']
    state_dict = {}
    
    for k, v in raw_state_dict.items():
      if k.startswith('module.'):
        k = k[7:]
      state_dict[k] = v
    model.load_state_dict(state_dict)
    
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

  else:
    kwargs = {'vocab'             : vocab,
              'image_size'        : config['image_size'],
              'embedding_dim'     : config['embedding_dim'],
              'gconv_dim'         : config['gconv_dim'],
              'gconv_hidden_dim'  : config['gconv_hidden_dim'],
              'gconv_num_layers'  : config['gconv_num_layers'],
              'mlp_normalization' : config['mlp_normalization'],
              'decoder_dims'      : config['decoder_network_dims'],
              'normalization'     : config['normalization'],
              'activation'        : config['activation'],
              'mask_size'         : config['mask_size'],
              'layout_noise_dim'  : config['layout_noise_dim'],
              'img_feats_branch'  : config['image_feats'],
              'feats_in_gcn'      : config['feats_in_gcn'],
              'feats_out_gcn'     : config['feats_out_gcn'],
              'is_baseline'       : config['is_baseline'],
              'is_supervised'     : config['is_supervised'],
              'spade_blocks'      : config['spade_gen_blocks'],
              'layout_pooling'    : config['layout_pooling']
            }

    model = SIMSGModel(**kwargs)

  return model, kwargs


def build_obj_discriminator(config, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = config['discriminator_loss_weight']
  d_obj_weight = config['d_obj_weight']
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'vocab': vocab,
    'arch': config['d_obj_arch'],
    'normalization': config['d_normalization'],
    'activation': config['d_activation'],
    'padding': config['d_padding'],
    'object_size': config['crop_size'],
  }
  discriminator = AcCropDiscriminator(**d_kwargs)

  return discriminator, d_kwargs


def build_img_discriminator(config, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = config['discriminator_loss_weight']
  d_img_weight = config['d_img_weight']
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': config['d_img_arch'],
    'normalization': config['d_normalization'],
    'activation': config['d_activation'],
    'padding': config['d_padding'],
  }

  if config['multi_discriminator']:
    discriminator = MultiscaleDiscriminator(input_nc=3, num_D=2)
  else:
    discriminator = PatchDiscriminator(**d_kwargs)

  return discriminator, d_kwargs


def check_model(config, t, loader, model):

  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      imgs_src = None

      if config['dataset'] == "vg" or (config['dataset'] == "clevr" and not config['is_supervised']):
        imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = batch
      elif config['dataset'] == "clevr":
        imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
        triple_to_img, imgs_in = batch

      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks, src_image=imgs_in, imgs_src=imgs_src)
      imgs_pred, boxes_pred, masks_pred, _, _ = model_out

      skip_pixel_loss = False
      total_loss, losses = calculate_model_losses(config, skip_pixel_loss, imgs, imgs_pred, boxes, boxes_pred)

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= config['num_val_samples']:
        break

    samples = {}
    samples['gt_img'] = imgs

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in, imgs_src=imgs_src)
    samples['gt_box_gt_mask'] = model_out[0]

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, src_image=imgs_in, imgs_src=imgs_src)
    samples['generated_img_gt_box'] = model_out[0]
    samples['masked_img'] = model_out[3][:,:3,:,:]

    for k, v in samples.items():
      samples[k] = imagenet_deprocess_batch(v)

    mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    avg_iou = total_iou / total_boxes

    masks_to_store = masks
    if masks_to_store is not None:
      masks_to_store = masks_to_store.data.cpu().clone()

    masks_pred_to_store = masks_pred
    if masks_pred_to_store is not None:
      masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  batch_data = {
    'objs': objs.detach().cpu().clone(),
    'boxes_gt': boxes.detach().cpu().clone(),
    'masks_gt': masks_to_store,
    'triples': triples.detach().cpu().clone(),
    'obj_to_img': obj_to_img.detach().cpu().clone(),
    'triple_to_img': triple_to_img.detach().cpu().clone(),
    'boxes_pred': boxes_pred.detach().cpu().clone(),
    'masks_pred': masks_pred_to_store
  }
  out = [mean_losses, samples, batch_data, avg_iou]
  return tuple(out)


def main(config):
  # print(config)
  # check_args(args)

  float_dtype = torch.cuda.FloatTensor

  writer = SummaryWriter(config['log_dir']) if config['log_dir'] is not None else None

  vocab, train_loader, val_loader = build_train_loaders(config)
  model, model_kwargs = build_model(config, vocab)
  model.type(float_dtype)
  print(model)
  
  # ================================================  
  print(f"Using {torch.cuda.device_count()} GPUs!")
  # ================================================

  # use to freeze parts of the network (VGG feature extraction)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

  obj_discriminator, d_obj_kwargs = build_obj_discriminator(config, vocab)
  img_discriminator, d_img_kwargs = build_img_discriminator(config, vocab)

  gan_g_loss, gan_d_loss = get_gan_losses(config['gan_loss_type'])

  if obj_discriminator is not None:
    obj_discriminator.type(float_dtype)
    obj_discriminator.train()
    print(obj_discriminator)
    optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(), lr=config['learning_rate'])

  if img_discriminator is not None:
    img_discriminator.type(float_dtype)
    img_discriminator.train()
    print(img_discriminator)

    optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(), lr= config['learning_rate'])

  restore_path = None
  if config['checkpoint_start_from'] is not None:
    restore_path = config['checkpoint_start_from']
  else:
    if config['restore_from_checkpoint']:
      restore_path = '%s_model.pt' % config['checkpoint_name']
      restore_path = os.path.join(config['output_dir'], restore_path)
  if restore_path is not None and os.path.isfile(restore_path):
    print('Restoring from checkpoint:')
    print(restore_path)
    checkpoint = torch.load(restore_path)

    model.load_state_dict(checkpoint['model_state'], strict=False)
    # print(optimizer)
    # optimizer.load_state_dict(checkpoint['optim_state'])

    if obj_discriminator is not None:
      obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
      optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

    if img_discriminator is not None:
      img_discriminator.load_state_dict(checkpoint['d_img_state'])
      optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

    t = checkpoint['counters']['t']
    print(t, config['eval_mode_after'])
    if 0 <= config['eval_mode_after'] <= t:
      model.eval()
    else:
      model.train()
    epoch = checkpoint['counters']['epoch']
  else:
    t, epoch = 0, 0
    checkpoint = init_checkpoint_dict(config, vocab, model_kwargs, d_obj_kwargs, d_img_kwargs)

  while True:
    if t >= config['num_iterations']:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)

    for batch in tqdm.tqdm(train_loader):
      if t == config['eval_mode_after']:
        print('switching to eval mode')
        model.eval()
        # filter to freeze feats net
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      imgs_src = None

      if config['dataset'] == "vg" or (config['dataset'] == "clevr" and not config['is_supervised']):
        imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = batch
      elif config['dataset'] == "clevr":
        imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
        triple_to_img, imgs_in = batch

      with timeit('forward', config['timing']):
        model_boxes = boxes
        model_masks = masks

        model_out = model(objs, triples, obj_to_img, boxes_gt=model_boxes, masks_gt=model_masks, src_image=imgs_in, imgs_src=imgs_src, t=t)
        imgs_pred, boxes_pred, masks_pred, layout_mask, _ = model_out

      with timeit('loss', config['timing']):
        # Skip the pixel loss if not using GT boxes
        skip_pixel_loss = (model_boxes is None)
        total_loss, losses = calculate_model_losses(config, skip_pixel_loss, imgs, imgs_pred, boxes, boxes_pred)

      if obj_discriminator is not None:

        obj_discr_ids = model_out[4]

        if obj_discr_ids is not None:
          if config['selective_discr_obj'] and torch.sum(obj_discr_ids) > 0:
            objs_ = objs[obj_discr_ids]
            boxes_ = boxes[obj_discr_ids]
            obj_to_img_ = obj_to_img[obj_discr_ids]

          else:
            objs_ = objs
            boxes_ = boxes
            obj_to_img_ = obj_to_img
        else:
          objs_ = objs
          boxes_ = boxes
          obj_to_img_ = obj_to_img

        scores_fake, ac_loss, layers_fake_obj = obj_discriminator(imgs_pred, objs_, boxes_, obj_to_img_)

        total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss',
                              config['ac_loss_weight'])
        weight = config['discriminator_loss_weight'] * config['d_obj_weight']
        total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses, 'g_gan_obj_loss', weight)

      if img_discriminator is not None:
        if not config['multi_discriminator']:
          scores_fake, layers_fake = img_discriminator(imgs_pred)

          weight = config['discriminator_loss_weight'] * config['d_img_weight']
          total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses, 'g_gan_img_loss', weight)
          if config['weight_gan_feat'] != 0:

            _, layers_real = img_discriminator(imgs)
            total_loss = add_loss(total_loss, gan_percept_loss(layers_real, layers_fake), losses, 'g_gan_percept_img_loss', weight * 10)
        else:
          fake_and_real = torch.cat([imgs_pred, imgs], dim=0)
          discriminator_out = img_discriminator(fake_and_real)
          scores_fake, scores_real = divide_pred(discriminator_out)

          weight = config['discriminator_loss_weight'] * config['d_img_weight']
          criterionGAN = GANLoss()
          img_g_loss = criterionGAN(scores_fake, True, for_discriminator=False)
          total_loss = add_loss(total_loss, img_g_loss, losses, 'g_gan_img_loss', weight)

          if config['weight_gan_feat'] != 0:
            criterionFeat = torch.nn.L1Loss()

            num_D = len(scores_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(scores_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterionFeat(
                        scores_fake[i][j], scores_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * config['weight_gan_feat'] / num_D
            total_loss = add_loss(total_loss, GAN_Feat_loss, losses, 'g_gan_feat_loss', 1.0)

          if config['percept_weight'] != 0:
            criterionVGG = VGGLoss()
            percept_loss = criterionVGG(imgs_pred, imgs)
            total_loss = add_loss(total_loss, percept_loss, losses, 'g_VGG', config['percept_weight'])

      losses['total_loss'] = total_loss.item()
      if not math.isfinite(losses['total_loss']):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      optimizer.zero_grad()
      with timeit('backward', config['timing']):
        total_loss.backward()
      optimizer.step()

      if obj_discriminator is not None:
        d_obj_losses = LossManager()
        imgs_fake = imgs_pred.detach()

        obj_discr_ids = model_out[4]

        if obj_discr_ids is not None:
          if config['selective_discr_obj'] and torch.sum(obj_discr_ids) > 0:

            objs_ = objs[obj_discr_ids]
            boxes_ = boxes[obj_discr_ids]
            obj_to_img_ = obj_to_img[obj_discr_ids]

          else:
            objs_ = objs
            boxes_ = boxes
            obj_to_img_ = obj_to_img
        else:
          objs_ = objs
          boxes_ = boxes
          obj_to_img_ = obj_to_img

        scores_fake, ac_loss_fake, _ = obj_discriminator(imgs_fake, objs_, boxes_, obj_to_img_)
        scores_real, ac_loss_real, _ = obj_discriminator(imgs, objs_, boxes_, obj_to_img_)

        d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
        d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
        d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
        d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

        optimizer_d_obj.zero_grad()
        d_obj_losses.total_loss.backward()
        optimizer_d_obj.step()

      if img_discriminator is not None:
        d_img_losses = LossManager()
        imgs_fake = imgs_pred.detach()

        if not config['multi_discriminator']:

          scores_fake = img_discriminator(imgs_fake)
          scores_real = img_discriminator(imgs)

          d_img_gan_loss = gan_d_loss(scores_real[0], scores_fake[0])
          d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

        else:

          fake_and_real = torch.cat([imgs_fake, imgs], dim=0)
          discriminator_out = img_discriminator(fake_and_real)
          scores_fake, scores_real = divide_pred(discriminator_out)

          d_img_gan_loss = criterionGAN(scores_fake, False, for_discriminator=True) \
                           + criterionGAN(scores_real, True, for_discriminator=True)

          d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

        optimizer_d_img.zero_grad()
        d_img_losses.total_loss.backward()
        optimizer_d_img.step()

      if t % config['print_every'] == 0:

        print_G_state(config, t, losses, writer, checkpoint)
        if obj_discriminator is not None:
          print_D_obj_state(config, t, writer, checkpoint, d_obj_losses)
        if img_discriminator is not None:
          print_D_img_state(config, t, writer, checkpoint, d_img_losses)

      if t % config['checkpoint_every'] == 0:
        print('checking on train')
        train_results = check_model(config, t, train_loader, model)
        t_losses, t_samples, t_batch_data, t_avg_iou = train_results

        checkpoint['checkpoint_ts'].append(t)
        checkpoint['train_iou'].append(t_avg_iou)

        print('checking on val')
        val_results = check_model(config, t, val_loader, model)
        val_losses, val_samples, val_batch_data, val_avg_iou = val_results

        checkpoint['val_iou'].append(val_avg_iou)

        # write images to tensorboard
        train_samples_viz = torch.cat((t_samples['gt_img'][:config['max_num_imgs'], :, :, :],
                                       t_samples['masked_img'][:config['max_num_imgs'], :, :, :],
                                       t_samples['generated_img_gt_box'][:config['max_num_imgs'], :, :, :]), dim=3)

        val_samples_viz = torch.cat((val_samples['gt_img'][:config['max_num_imgs'], :, :, :],
                                     val_samples['masked_img'][:config['max_num_imgs'], :, :, :],
                                     val_samples['generated_img_gt_box'][:config['max_num_imgs'], :, :, :]), dim=3)

        writer.add_image('Train samples', make_grid(train_samples_viz, nrow=4, padding=4), global_step=t)
        writer.add_image('Val samples', make_grid(val_samples_viz, nrow=4, padding=4), global_step=t)

        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)
        # write IoU to tensorboard
        writer.add_scalar('train mIoU', t_avg_iou, global_step=t)
        writer.add_scalar('val mIoU', val_avg_iou, global_step=t)
        # write losses to tensorboard
        for k, v in t_losses.items():
          writer.add_scalar('Train {}'.format(k), v, global_step=t)

        for k, v in val_losses.items():
          checkpoint['val_losses'][k].append(v)
          writer.add_scalar('Val {}'.format(k), v, global_step=t)
        checkpoint['model_state'] = model.state_dict()

        if obj_discriminator is not None:
          checkpoint['d_obj_state'] = obj_discriminator.state_dict()
          checkpoint['d_obj_optim_state'] = optimizer_d_obj.state_dict()

        if img_discriminator is not None:
          checkpoint['d_img_state'] = img_discriminator.state_dict()
          checkpoint['d_img_optim_state'] = optimizer_d_img.state_dict()

        checkpoint['optim_state'] = optimizer.state_dict()
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint_path_step = os.path.join(config['output_dir'], '%s_%s_model.pt' % (config['checkpoint_name'], str(t//10000)))
        checkpoint_path_latest = os.path.join(config['output_dir'], '%s_model.pt' % (config['checkpoint_name']))

        print('Saving checkpoint to ', checkpoint_path_latest)
        torch.save(checkpoint, checkpoint_path_latest)
        if t % 10000 == 0 and t >= 100000:
          torch.save(checkpoint, checkpoint_path_step)

if __name__ == '__main__':
  main()
