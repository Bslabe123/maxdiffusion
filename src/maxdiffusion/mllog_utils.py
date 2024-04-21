"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Utils that relevant to mllog for mlperf submission compliance."""
from typing import Optional
import jax
from mlperf_logging import mllog
import numpy as np

mllogger = mllog.get_mllogger()

def train_init_start():
  if jax.process_index() == 0:
    mllogger.event(mllog.constants.CACHE_CLEAR)
    mllogger.start(mllog.constants.INIT_START)

def train_init_stop():
  if jax.process_index() == 0:
    mllogger.end(mllog.constants.INIT_STOP)

def train_run_start():
  if jax.process_index() == 0:
    mllogger.start(mllog.constants.RUN_START)

def train_run_end():
  if jax.process_index() == 0:
    mllogger.end(mllog.constants.RUN_STOP, metadata={'status': 'success'})

def train_init_print(config, device: str = 'tpu-v5p'):
  """an initial mllog for mlperf sumbission compliance check."""
  if jax.process_index() == 0:
    mllogger.event(mllog.constants.SUBMISSION_ORG, 'Google')
    mllogger.event(mllog.constants.SUBMISSION_PLATFORM, device)
    mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
    mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)
    mllogger.event(mllog.constants.SUBMISSION_BENCHMARK, mllog.constants.STABLE_DIFFUSION)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, 1)
    mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE,
                   config.per_device_batch_size * jax.device_count())

    mllogger.event(mllog.constants.OPT_NAME, mllog.constants.ADAMW)
    mllogger.event(mllog.constants.OPT_ADAMW_BETA_1, config.adam_b1)
    mllogger.event(mllog.constants.OPT_ADAMW_BETA_2, config.adam_b2)
    mllogger.event(mllog.constants.OPT_ADAMW_EPSILON, config.adam_eps)
    mllogger.event(mllog.constants.OPT_ADAMW_WEIGHT_DECAY, config.adam_weight_decay)

    mllogger.event(mllog.constants.OPT_BASE_LR, config.learning_rate)
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS,
                   int(config.learning_rate_schedule_steps * config.warmup_steps_fraction))

    # Training: a subset of laion-400m
    # Validation: a subset of coco-2014 validation
    mllogger.event(mllog.constants.TRAIN_SAMPLES, 6513144)
    mllogger.event(mllog.constants.EVAL_SAMPLES, 30000)

    mllogger.event(mllog.constants.SEED, config.seed)

def train_step_start(step_num):
  if jax.process_index() == 0:
    mllogger.start(
      mllog.constants.BLOCK_START,
      value="training_step",
      metadata={
        mllog.constants.STEP_NUM: step_num,
      },
    )

def train_step_end(step_num, loss, lr):
  if jax.process_index() == 0:
    mllogger.end(
      mllog.constants.BLOCK_STOP,
      value="training_step",
      metadata={
        mllog.constants.STEP_NUM: step_num,
        'loss': loss,
        'lr': lr,
      },
    )

def maybe_train_step_log(config, start_step, step, metric, train_log_interval: int = 100):
  if step > start_step and step % train_log_interval == 0 or step == config.max_train_steps - 1:
    # convert the jax array to a numpy array for mllog JSON encoding
    loss = np.asarray(metric['scalar']['learning/loss'])
    lr = np.asarray(metric['scalar']['learning/current_learning_rate'])

    train_step_end(step, loss, lr)
    # start new tracking except the last step
    if step < config.max_train_steps - 1:
      train_step_start(step)

def train_checkpoint_step_log(config, step_num: int):
    mllogger.event(
      "checkpoint",
      value=step_num,
      metadata={
        mllog.constants.STEP_NUM: step_num,
      },
    )

def extract_step_from_ckpt_name(model_ckpt_name: str) -> int:
  # model_ckpt_name format:
  #    {path/to/ckpt}/checkpoints/{step_num}
  step_num_str = model_ckpt_name.strip('/').split('/')[-1]
  step_num = int(step_num_str)
  return step_num

def eval_start(config):
  if jax.process_index() == 0:
    step_num = extract_step_from_ckpt_name(config.pretrained_model_name_or_path)
    mllogger.start(
      mllog.constants.EVAL_START,
      metadata={
        mllog.constants.STEP_NUM: step_num,
      },
    )

def eval_end(config):
  if jax.process_index() == 0:
    step_num = extract_step_from_ckpt_name(config.pretrained_model_name_or_path)
    mllogger.end(
      mllog.constants.EVAL_STOP,
      metadata={
        'step_num': step_num,
      },
    )

def eval_fid(config, fid: float):
  if jax.process_index() == 0:
    step_num = extract_step_from_ckpt_name(config.pretrained_model_name_or_path)
    mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      value=fid,
      metadata={
        mllog.constants.STEP_NUM: step_num,
        "metric": "FID",
        "ckpt_name": config.pretrained_model_name_or_path,
      },
    )

def eval_clip(config, clip_score: float):
  if jax.process_index() == 0:
    step_num = extract_step_from_ckpt_name(config.pretrained_model_name_or_path)
    mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      value=clip_score,
      metadata={
        mllog.constants.STEP_NUM: step_num,
        "metric": "CLIP",
        "ckpt_name": config.pretrained_model_name_or_path,
      },
    )

def timestamp_fid(fid: float, timestamp: int, step_num: int):
  if jax.process_index() == 0:
    mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      value=fid,
      metadata={
        mllog.constants.STEP_NUM: step_num,
        "metric": "FID",
      },
      time_ms=timestamp,
    )

def timestamp_clip(clip: float, timestamp: int, step_num: int):
  if jax.process_index() == 0:
    mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      value=clip,
      metadata={
        mllog.constants.STEP_NUM: step_num,
        "metric": "CLIP",
      },
      time_ms=timestamp,
    )

def timestamp_run_stop_success(step_num: int, timestamp: int):
  if jax.process_index() == 0:
      assert step_num
      assert timestamp
      mllogger.end(
        mllog.constants.RUN_STOP,
        metadata={
          mllog.constants.STATUS: mllog.constants.SUCCESS,
          mllog.constants.STEP_NUM: step_num,
        },
        time_ms=timestamp,
      )

def timestamp_run_stop_abort():
  if jax.process_index() == 0:
      mllogger.end(
        mllog.constants.RUN_STOP,
        metadata={
          mllog.constants.STATUS: mllog.constants.ABORTED,
        },
      )