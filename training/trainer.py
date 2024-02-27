# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import time
from collections.abc import Mapping
from distutils.util import strtobool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import math
import gc

from dataclasses import dataclass, field
from datasets import Dataset
from transformers import Trainer as HFTrainer
# Integrations must be imported before ML frameworks:

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR


from transformers import __version__
from transformers.trainer_callback import (
    PrinterCallback,
    TrainerCallback,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    get_parameter_names,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import (
    get_full_repo_name,
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.optimization import get_scheduler
from transformers import TrainingArguments as HfTrainingArguments

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm # type: ignore
    import torch_xla.distributed.parallel_loader as pl # type: ignore


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp # type: ignore
    from smdistributed.modelparallel import __version__ as SMP_VERSION  # type: ignore

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


from transformers.trainer import logger


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class LogCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_time_interval = 0
        self.current_step = 0
        self.is_training = False
        self.max_steps = -1
        self.first_step_of_run = 0

    def on_train_begin(self, args, state, control, **kwargs):
        args.logging_steps = 1
        args.logging_strategy = "steps"
        if state.is_local_process_zero:
            self.log_time_interval = getattr(args, "log_time_interval", 0)
            if self.log_time_interval > 0:
                logger.info(f"Using log_time_interval {self.log_time_interval} s. This will override logging_steps and logging_strategy.")
            self.is_training = True
            self.current_step = 0
            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.max_steps = state.max_steps
            self.first_step_of_run = state.global_step
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if self.is_training:
                current_time = time.time()
                time_diff = current_time - self.last_log_time
                force = logs.get("force", False)
                if time_diff > self.log_time_interval or self.current_step >= self.max_steps - 1 or force:
                    self.last_log_time = current_time
                    steps_completed = max(self.current_step, 1)
                    steps_since_first = max(1, self.current_step - self.first_step_of_run)
                    remaining_steps = self.max_steps - steps_completed
                    pct_completed = (steps_completed / self.max_steps) * 100
                    time_since_start = current_time - self.start_time
                    remaining_time = (time_since_start / steps_since_first) * remaining_steps
                    update = {'completed': f'{pct_completed:.2f}% ({steps_completed:_} / {self.max_steps:_})', 'remaining time': self.format_duration(remaining_time)}
                    logger.info(str({**logs, **update}))
            else:
                logger.info(str(logs))

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.current_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.is_training = False

    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{int(hours)}:{int(minutes):02}:{int(seconds):02}'

@dataclass
class TrainingArguments(HfTrainingArguments):
    min_lr_ratio: float = field(
        default=0.0
    )
    ordered: bool = field(
        default=False
    )
    cuda_empty_cache: bool = field(
        default=False, metadata={"help": "Empty cuda cache before every step."}
    )


def min_lr_bound(current_step: int, wrapped_func: Callable[[float], float], min_lr_ratio: float, warmup_steps: int):
    if current_step < warmup_steps:
        return wrapped_func(current_step)
    return min_lr_ratio + wrapped_func(current_step) * (1.0 - min_lr_ratio)


# - Callbacks: transformers.trainer_callback.DefaultFlowCallback, transformers.integrations.WandbCallback, transformers.trainer_callback.ProgressCallback
class Trainer(HFTrainer):
    def __init__(self, model, args, *more_args, **kwargs):
        super().__init__(model, args, *more_args, **kwargs)

        try:
            self.remove_callback(PrinterCallback)
            self.add_callback(LogCallback)
        except ValueError:
            logger.warn("Couldn't remove PrinterCallback")

    def compute_loss(self, model, inputs, return_outputs=False, return_output_and_metrics=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # print(torch.distributed.get_rank(), "-----"*100, flush=True)
        # print(torch.distributed.get_rank(), inputs["input_ids"][:,:10], flush=True)
        outputs = model(**inputs, use_cache=False)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if return_output_and_metrics:
            shifted_labels = inputs["labels"][:,1:].contiguous()
            valid_mask = (shifted_labels != -100)
            correct = (outputs.logits[:,:-1].argmax(-1) == shifted_labels).float()
            correct[~valid_mask] = 0.0
            acc = correct.sum(dim=-1) / valid_mask.float().sum(dim=-1)

            metrics = {"acc": acc}

            return (loss, outputs, metrics)
        if return_outputs:
            return (loss, outputs)
        else:
            return loss

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.min_lr_ratio != 0.0:
            if isinstance(self.lr_scheduler, LambdaLR):
                lr_lambdas = self.lr_scheduler.lr_lambdas
                new_lr_lambdas = [
                    lr_lambda
                    if lr_lambda is None or isinstance(lr_lambda, partial) and lr_lambda.func == min_lr_bound
                    else
                    partial(min_lr_bound,
                            wrapped_func=lr_lambda,
                            min_lr_ratio=self.args.min_lr_ratio,
                            warmup_steps=self.args.get_warmup_steps(num_training_steps))
                    for lr_lambda in lr_lambdas
                ]

                self.lr_scheduler.lr_lambdas = new_lr_lambdas
            else:
                raise NotImplementedError("Only LambdaLR is supported for min_lr_ratio")

        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        if self.args.ordered:
            return SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise ValueError("SageMaker Model Parallelism is not supported in BaseTrainer")
            else:
                with self.compute_loss_context_manager():
                    loss, outputs, metrics = self.compute_loss(model, inputs, return_output_and_metrics=True)
                if loss is not None:
                    loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]

        if prediction_loss_only:
            return (loss, None, None, metrics)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels, metrics)

    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        if self.args.cuda_empty_cache:
            gc.collect()
            torch.cuda.empty_cache()
        return self.autocast_smart_context_manager()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        metrics_host = None

        metrics_names = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_metrics = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, metrics = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()


            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if metrics is not None:
                if metrics_names is None:
                    metrics_names = list(metrics.keys())
                else:
                    assert metrics_names == list(metrics.keys()), "Metrics should have the same keys across batches"


                metrics = [
                    metric if metric.shape else metric.repeat(batch_size) for metric in metrics.values()
                ]
                metrics = self.accelerator.pad_across_processes(metrics, dim=1, pad_index=float('nan'))
                metrics = self.accelerator.gather_for_metrics(metrics)
                metrics_host = metrics if metrics_host is None else nested_concat(metrics_host, metrics, padding_index=float('nan'))
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)


            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if metrics_host is not None:
                    metrics = nested_numpify(metrics_host)
                    all_metrics = (
                        metrics if all_metrics is None else nested_concat(all_metrics, metrics, padding_index=float('nan'))
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if metrics_host is not None:
            metrics = nested_numpify(metrics_host)
            all_metrics = (
                metrics if all_metrics is None else nested_concat(all_metrics, metrics, padding_index=float('nan'))
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        # if all_metrics is not None:
        #     all_metrics = nested_truncate(all_metrics, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_metrics is not None:
            for key, value in zip(metrics_names, all_metrics):
                valid = ~np.isnan(value)
                metrics[key] = value[valid].mean().item()
                metrics[f"{key}___samples"] = np.sum(valid).item()

        metrics["samples"] = num_samples

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def evaluate(
        self,
        eval_dataset: Optional[Union[Dict[str, Dataset], Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if isinstance(eval_dataset, dict):
            metrics = {}
            for key, dataset in eval_dataset.items():
                metrics.update(super().evaluate(dataset, ignore_keys=ignore_keys, metric_key_prefix=f"{metric_key_prefix}_{key}"))
        else:
            metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        return metrics
