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

Description   : Script to run the train file for spade model 
Authors       : Jitong Ding
Date Created  : Apr 11' 2021
Data Modified : Apr 29' 2021

'''
import os, sys
import random, yaml
import shutil
import torch
from scripts.train import main as train_main



config = {  "seed": 1,
            "gpu": 0,
            "dataset": "clevr",
            "vg_image_dir": "./datasets/clevr/target",
            "output_dir": "experiments/clevr",
            "checkpoint_name": "spade_64_clevr",
            "log_dir": "experiments/clevr/logs/spade_64_clevr",
            "image_size": (64, 64), # (128, 128)
            "crop_size": 32,    
            "batch_size": 32,       # 4
            "mask_size": 16,
            "d_obj_arch": "C4-64-2,C4-128-2,C4-256-2",
            "d_img_arch": "C4-64-2,C4-128-2,C4-256-2",
            "decoder_network_dims": (1024,512,256,128,64),
            "layout_pooling": "sum",
            "percept_weight" : 0,
            "weight_gan_feat": 0,
            "discriminator_loss_weight": 0.01,
            "d_obj_weight": 1,
            "ac_loss_weight": 0.1,
            "d_img_weight": 1,
            "l1_pixel_loss_weight": 1,
            "bbox_pred_loss_weight": 10,
            "feats_in_gcn": True,
            "feats_out_gcn": True,
            "is_baseline": False,
            "is_supervised": False,
            "num_iterations": 50000,
            "print_every": 500,
            "checkpoint_every": 2000,
            "max_num_imgs": 32,
            "vocab_json": "./datasets/clevr/target/vocab.json",
            "train_h5": "./datasets/clevr/target/train.h5",
            "val_h5": "./datasets/clevr/target/val.h5",
            "test_h5": "./datasets/clevr/target/test.h5",
            "max_objects_per_image": 10,
            "vg_use_orphaned_objects": True,
            "embedding_dim": 128,
            "gconv_dim": 256,
            "gconv_hidden_dim": 512,
            "gconv_num_layers": 5,
            "mlp_normalization": None,
            "normalization": "batch",
            "activation": "leakyrelu-0.2",
            "layout_noise_dim": 32,
            "image_feats": True,
            "selective_discr_obj": True,
            "gan_loss_type": 'gan',
            "d_normalization": 'batch',
            "d_padding": 'valid',
            "d_activation": "leakyrelu-0.2",
            "timing" : False,
            "checkpoint_start_from": None,
            "restore_from_checkpoint": True,
            "multi_discriminator" : False,
            "spade_gen_blocks" : False,
            "layout_pooling" : "sum",
            "num_train_samples": None,
            "include_relationships": True,
            "loader_num_workers": 4,
            "shuffle_val": True,
            "learning_rate": 2e-4,
            "eval_mode_after": 100000,
        }

def main():

    if torch.cuda.is_available():
        torch.cuda.set_device(config['gpu'])
        torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    train_main(config)

if __name__ == '__main__':
    main()
