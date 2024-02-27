from itertools import combinations
import logging
import os
import random
import sys
import torch
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from modeling.modeling_flash_llama import LlamaForSequenceClassification

from training.trainer import Trainer, TrainingArguments as BaseTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from typing import Any, Dict
from dataclasses import dataclass, field
from typing import Optional, List
import functools

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ScriptArguments:
    text_field: str = field(
        default="texts",
        metadata={
            "help": "Name of the text field in the dataset."
        },
    )
    label_field: List[str] = field(
        default="calibrated_predictions",
        metadata={
            "help": "Name of the label field in the dataset."
        },
    )
    train_datasets: List[str] = field(
        default=None,
        metadata={"help": "Path to training dataset."},
    )
    eval_datasets: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to eval dataset."},
    )
    eval_split_size: float = field(
        default=0.0,
        metadata={"help": "Validation split size."},
    )
    eval_split_size_train: Optional[float] = field(
        default=None,
        metadata={"help": "Validation split size for training datasets."},
    )

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    lora: bool = field(default=False, metadata={"help": "Whether to use parameter efficient fine-tuning."})
    lora_path: str = field(default=None, metadata={"help": "Path to the lora model."})
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_target_modules: List[str] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})

    single_label_ablation: int = field(default=-1, metadata={"help": "Whether to use single label for ablation."})

@dataclass
class TrainingArguments(BaseTrainingArguments):
    label_temperature: float = field(
        default=1.0,
        metadata={"help": "Label temperature"},
    )
    log_confidences: List[float] = field(
        default_factory=lambda: [0.5, 0.8],
        metadata={"help": "Confidence thresholds for logging accuracy"},
    )
    confidence_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Confidence threshold for including data during training"
        },
    )

class DataCollator:
    def __init__(self, args, training_args, tokenizer):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_length = self.args.max_length

    @torch.no_grad()
    def __call__(self, features: Any) -> Dict[str, Any]:
        batch = self.tokenizer(
            sum([item[self.args.text_field] for item in features], []),
            add_special_tokens=False, truncation=True, return_tensors="pt", padding=True, max_length=self.max_length)

        bsz = batch.input_ids.size(0)
        num_labels = len(self.args.label_field)
        labels = -100 * torch.ones(bsz, bsz, num_labels, dtype=torch.float32)

        counter = 0
        for item in features:
            k = len(item[self.args.text_field])
            for i, label in enumerate(self.args.label_field):
                labels[counter:counter + k, counter:counter + k, i] = torch.tensor(item[label], dtype=torch.float32)
            counter += k

        for i in range(labels.size(-1)):
            if self.args.single_label_ablation >= 0 and i != self.args.single_label_ablation:
                labels[:,:,i] = -100

        labels[range(bsz), range(bsz)] = -100

        return dict(input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=labels)


def confidence_mask(labels, confidence):
    return (labels - 0.5).abs() >= confidence / 2


class LabelFilter:
    def __init__(self, label_field):
        self.label_field = label_field

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return not (labels == -100).all().item()


class ConfidenceFilter:
    def __init__(self, label_field, confidence):
        self.label_field = label_field
        self.confidence = confidence

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return confidence_mask(labels[labels != -100], self.confidence).any().item()


def bce_with_temperature(probs, labels, temperature = 1.0):
    probs = probs.clamp(min=0.0, max=1.0)
    labels = labels.clamp(min=0.0, max=1.0)

    if temperature != 1.0:
        labels = (labels.logit() / temperature).sigmoid()

    return torch.nn.functional.binary_cross_entropy(probs, labels)


class PreferenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, return_output_and_metrics=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logit_diffs = outputs.logits.unsqueeze(0) - outputs.logits.unsqueeze(1)

        probs = logit_diffs.float().sigmoid()

        valid_mask = (labels != -100) & confidence_mask(labels, self.args.confidence_threshold)
        loss = bce_with_temperature(probs[valid_mask], labels[valid_mask], self.args.label_temperature)

        if return_output_and_metrics:
            correct = torch.where(
                (labels != -100), ((probs >= 0.5) == (labels >= 0.5)).float(), float('nan')
            )

            metrics = {
                "acc": correct,
            }
            for confidence in self.args.log_confidences:
                metrics[f"acc_confidence{confidence * 100}"] = torch.where(confidence_mask(labels, confidence), correct, float('nan'))

            for i in range(probs.shape[-1]):
                valid_mask = (labels[..., i] != -100)
                loss_i = bce_with_temperature(probs[..., i][valid_mask], labels[..., i][valid_mask], self.args.label_temperature)

                # wandb: eval/arxiv_self_label3_acc___samples       1428
                # wandb: eval/arxiv_self_label3_acc_confidence50.0  0.8758

                # valid_mask = (labels[..., i] != -100)
                valid_mask = (labels[..., i] != -100) & torch.triu(torch.ones_like(labels[..., i])).bool()
                correct = torch.where(
                    valid_mask, ((probs[..., i] >= 0.5) == (labels[..., i] >= 0.5)).float(), float('nan')
                )

                baseline_correct = (labels[..., i] >= 0.5).float()
                baseline_correct = torch.where(valid_mask, baseline_correct, float('nan'))

                metrics.update({
                    f"label{i}_loss": loss_i,
                    f"label{i}_acc": correct,
                    # f"label{i}_baseline_acc": baseline_correct,
                })
                for confidence in self.args.log_confidences:
                    metrics.update({
                        f"label{i}_acc_confidence{confidence * 100}": torch.where(confidence_mask(labels[..., i], confidence), correct, float('nan')),
                        # f"label{i}_baseline_acc_confidence{confidence * 100}": torch.where(confidence_mask(labels[..., i], confidence), baseline_correct, float('nan')),
                    })

            return (loss, outputs, metrics)
        if return_outputs:
            return (loss, outputs)
        else:
            return loss


def load_datasets(dataset_paths, eval_split_size, seed, label_field, num_workers, cache_dir):
    train_datasets = {}
    eval_datasets = {}

    loaded_datasets = {}
    for path in dataset_paths:
        if os.path.exists(path + "/state.json"):
            dataset = datasets.load_from_disk(path)
        else:
            dataset = datasets.load_dataset(path, cache_dir=cache_dir)

        if isinstance(dataset, datasets.DatasetDict):
            if "train" in dataset and len(dataset) == 1:
                loaded_datasets[path] = dataset["train"]
            else:
                for split, ds in dataset.items():
                    loaded_datasets[f"{path}/{split}"] = ds
        else:
            loaded_datasets[path] = dataset

    for path, dataset in loaded_datasets.items():
        if "examples" in dataset.column_names:
            dataset = dataset.remove_columns("examples")
        dataset = dataset.filter(LabelFilter(label_field), num_proc=num_workers, keep_in_memory=True)

        if eval_split_size < 1.0:
            splits = dataset.train_test_split(test_size=eval_split_size, seed=seed)
            train_dataset, eval_dataset = splits["train"], splits["test"]
        else:
            eval_dataset = dataset
            train_dataset = dataset.select([])

        dataset_name = os.path.basename(path)

        if dataset_name in train_datasets:
            train_datasets[dataset_name] = (
                datasets.concatenate_datasets([train_datasets[dataset_name], train_dataset])
            )
            eval_datasets[dataset_name] = (
                datasets.concatenate_datasets([eval_datasets[dataset_name], eval_dataset])
            )
        else:
            train_datasets[dataset_name] = train_dataset
            eval_datasets[dataset_name] = eval_dataset
    return train_datasets, eval_datasets


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional arguments {args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    config = AutoConfig.from_pretrained(
        args.config_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None
    )
    if args.config_overrides:
        logger.info(f"Overriding config: {args.config_overrides}")
        config.update_from_string(args.config_overrides)
        logger.info(f"New config: {config}")

    config.num_labels = len(args.label_field)
    tokenizer.pad_token_id = 0
    config.pad_token_id = 0

    if args.model_name_or_path:
        half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
        model = LlamaForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            torch_dtype=(half_dtype if args.lora or args.lora_path else None),
        )
    else:
        model = AutoModelForSequenceClassification.from_config(config)

    if args.lora or args.lora_path:
        from peft import PeftModel, get_peft_model, LoraConfig, TaskType
        if args.lora_path:
            logger.info(f"Loading LoRA model from {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                modules_to_save=args.lora_modules_to_save,
            )
            model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    logger.info(f"Model: {model}")

    train_dataset = {}
    val_dataset = {}

    train_dataset, val_dataset = load_datasets(
        args.train_datasets,
        args.eval_split_size_train if args.eval_split_size_train is not None else args.eval_split_size,
        training_args.seed,
        args.label_field,
        training_args.dataloader_num_workers,
        cache_dir=args.cache_dir
    )
    train_dataset = datasets.concatenate_datasets(list(train_dataset.values()))
    val_dataset = datasets.concatenate_datasets(list(val_dataset.values()))

    logger.warning(f"Before confidence filtering - train sequences: {len(train_dataset):,} - validation sequences: {len(val_dataset):,}")
    train_dataset = train_dataset.filter(
        ConfidenceFilter(args.label_field, training_args.confidence_threshold),
        num_proc=training_args.dataloader_num_workers, keep_in_memory=True
    )
    val_dataset = val_dataset.filter(
        ConfidenceFilter(args.label_field, training_args.confidence_threshold),
        num_proc=training_args.dataloader_num_workers, keep_in_memory=True
    )
    logger.warning(f"After confidence filtering - train sequences: {len(train_dataset):,} - validation sequences: {len(val_dataset):,}")

    _, eval_dataset = load_datasets(
        args.eval_datasets,
        args.eval_split_size,
        training_args.seed,
        args.label_field,
        training_args.dataloader_num_workers,
        cache_dir=args.cache_dir
    )

    eval_dataset["all"] = datasets.concatenate_datasets(list(eval_dataset.values()))
    logger.warning(f"All eval sequences: {len(eval_dataset['all']):,}")

    eval_dataset["validation"] = val_dataset

    collator = DataCollator(args, training_args, tokenizer)

    # Initialize our Trainer
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    if trainer.is_fsdp_enabled:
        # Identify which modules have "layer" in their class name and use these
        # as the basic FSDP blocks that are sharded and exchanged between GPUs
        def layer_policy_fn(module):
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=layer_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
