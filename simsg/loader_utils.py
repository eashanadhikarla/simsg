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


from simsg.data.vg import SceneGraphNoPairsDataset, collate_fn_nopairs
from simsg.data.clevr import SceneGraphWithPairsDataset, collate_fn_withpairs

import json
from torch.utils.data import DataLoader


def build_clevr_supervised_train_dsets(args):
  print("building fully supervised %s dataset" % args.dataset)
  with open(args['vocab_json'], 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args['train_h5'],
    'image_dir': args['vg_image_dir'],
    'image_size': args['image_size'],
    'max_samples': args['num_train_samples'],
    'max_objects': args['max_objects_per_image'],
    'use_orphaned_objects': args['vg_use_orphaned_objects'],
    'include_relationships': args['include_relationships'],
  }
  train_dset = SceneGraphWithPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args['batch_size']
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args["val_h5"]
  del dset_kwargs['max_samples']
  val_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  dset_kwargs['h5_path'] = args["test_h5"]
  test_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset, test_dset


def build_dset_nopairs(args, checkpoint):

  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode,
    'predgraphs': args.predgraphs
  }
  dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return dset


def build_dset_withpairs(args, checkpoint, vocab_t):

  vocab = vocab_t
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode
  }
  dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return dset


def build_eval_loader(args, checkpoint, vocab_t=None, no_gt=False):

  if args.dataset == 'vg' or (no_gt and args.dataset == 'clevr'):
    dset = build_dset_nopairs(args, checkpoint)
    collate_fn = collate_fn_nopairs
  elif args.dataset == 'clevr':
    dset = build_dset_withpairs(args, checkpoint, vocab_t)
    collate_fn = collate_fn_withpairs

  loader_kwargs = {
    'batch_size': 1,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)

  return loader


def build_train_dsets(args):
  print("building unpaired %s dataset" % args["dataset"])
  with open(args["vocab_json"], 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args["train_h5"],
    'image_dir': args["vg_image_dir"],
    'image_size': args["image_size"],
    'max_samples': args["num_train_samples"],
    'max_objects': args["max_objects_per_image"],
    'use_orphaned_objects': args["vg_use_orphaned_objects"],
    'include_relationships': args["include_relationships"],
  }
  train_dset = SceneGraphNoPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args["batch_size"]
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args["val_h5"]
  del dset_kwargs['max_samples']
  val_dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset

def build_train_loaders(args):
  print(args["dataset"])
  if args["dataset"] == 'vg' or (args["dataset"] == "clevr" and not args["is_supervised"]):
    vocab, train_dset, val_dset = build_train_dsets(args)
    collate_fn = collate_fn_nopairs

  elif args["dataset"] == 'clevr':
    vocab, train_dset, val_dset, test_dset = build_clevr_supervised_train_dsets(args)
    collate_fn = collate_fn_withpairs
  loader_kwargs = {
    'batch_size': args["batch_size"],
    'num_workers': args["loader_num_workers"],
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  loader_kwargs['shuffle'] = args["shuffle_val"]
  val_loader = DataLoader(val_dset, **loader_kwargs)
  
  return vocab, train_loader, val_loader
