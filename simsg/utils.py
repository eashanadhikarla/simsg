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
Date Created  : Apr 11' 2021
Data Modified : Apr 29' 2021
Description   : Script to train SIMSG | CRN - 64 & SPADE - 64

'''


import time
import inspect
import subprocess
from contextlib import contextmanager

import torch


def int_tuple(s):
  return tuple(int(i) for i in s.split(','))


def float_tuple(s):
  return tuple(float(i) for i in s.split(','))


def str_tuple(s):
  return tuple(s.split(','))


def bool_flag(s):
  if s == '1' or s == 'True' or s == 'true':
    return True
  elif s == '0' or s == 'False' or s == 'false':
    return False
  msg = 'Invalid value "%s" for bool flag (should be 0/1 or True/False or true/false)'
  raise ValueError(msg % s)


def lineno():
  return inspect.currentframe().f_back.f_lineno


def get_gpu_memory():
  torch.cuda.synchronize()
  opts = [
      'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
  ]
  cmd = str.join(' ', opts)
  ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  output = ps.communicate()[0].decode('utf-8')
  output = output.split("\n")[1].split(":")
  consumed_mem = int(output[1].strip().split(" ")[0])
  return consumed_mem


@contextmanager
def timeit(msg, should_time=True):
  if should_time:
    torch.cuda.synchronize()
    t0 = time.time()
  yield
  if should_time:
    torch.cuda.synchronize()
    t1 = time.time()
    duration = (t1 - t0) * 1000.0
    print('%s: %.2f ms' % (msg, duration))


class LossManager(object):
  def __init__(self):
    self.total_loss = None
    self.all_losses = {}

  def add_loss(self, loss, name, weight=1.0):
    cur_loss = loss * weight
    if self.total_loss is not None:
      self.total_loss += cur_loss
    else:
      self.total_loss = cur_loss

    self.all_losses[name] = cur_loss.data.cpu().item()

  def items(self):
    return self.all_losses.items()

