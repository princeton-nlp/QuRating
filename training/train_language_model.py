import logging
import os
import sys
import torch
import datasets
import transformers
import math
import functools

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from modeling.modeling_flash_llama import LlamaForCausalLM
from training.trainer import Trainer, TrainingArguments

from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from transformers.trainer_utils import get_last_checkpoint
import json
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
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
    config_overrides_json: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "'{\"resid_pdrop\": 0.2}'"
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

    tokenized_train_dataset: List[str] = field(
        default=None, metadata={"help": "The path of the train datasets to use (via the datasets library)."}
    )
    tokenized_validation_dataset: Optional[str] = field(
        default=None, metadata={"help": "The path of the train dataset to use (via the datasets library)."}
    )
    tokenized_test_dataset: Optional[str] = field(
        default=None, metadata={"help": "The path of the train dataset to use (via the datasets library)."}
    )

    half_precision_training: bool = field(
        default=False,
        metadata={"help": "Whether to use full fp16/bf16 training."},
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

    infill_proportion: float = field(default=0.0, metadata={"help": "Proportion of sequences for infilling"})
    infill_rate_min: float = field(default=0.05, metadata={"help": "Min rate of infilling"})
    infill_rate_max: float = field(default=0.5, metadata={"help": "Max rate of infilling"})
    infill_mean_length_min: float = field(default=3, metadata={"help": "Min mean span length"})
    infill_mean_length_max: float = field(default=30, metadata={"help": "Max mean span length"})
    infill_random_order: bool = field(default=False, metadata={"help": "Whether to permute targets"})

    infill_ignore_run_in: int = field(default=0, metadata={"help": "Max mean span length"})

    sort_by: Optional[str] = field(
        default=None, metadata={"help": "The column to sort by."}
    )
    reverse_sort: bool = field(
        default=False, metadata={"help": "Whether to reverse the sort order."}
    )


class DataCollator:
    def __init__(self, args, training_args, tokenizer):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = 0
        self.pad_token_id = 0

        self.mean_span_length = torch.distributions.Uniform(args.infill_mean_length_min, args.infill_mean_length_max)
        self.infill_rate = torch.distributions.Uniform(args.infill_rate_min, args.infill_rate_max)

        if self.args.infill_proportion > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [f"<mask{num}>" for num in range(512)]})
            self.special_token_ids = torch.tensor(self.tokenizer.additional_special_tokens_ids, dtype=torch.long)

    def random_span_lengths(self, num_spans, num_tokens, dtype=None, device=None):
        span_starts = torch.randperm(num_tokens - 1, dtype=dtype, device=device)[:num_spans - 1] + 1
        span_starts = torch.sort(span_starts).values
        span_lengths = torch.diff(span_starts,
                                  prepend=torch.tensor([0], dtype=dtype, device=device),
                                  append=torch.tensor([num_tokens], dtype=dtype, device=device))
        return span_lengths

    @torch.no_grad()
    def span_masking(self, seq):
        mean_span_length = self.mean_span_length.sample().item()
        infill_rate = self.infill_rate.sample().item()

        N = len(seq)
        masks_per_seq = math.ceil(N * infill_rate)
        num_spans = max(round(masks_per_seq / mean_span_length), 1)

        unmasked_lengths = self.random_span_lengths(num_spans+1, N - masks_per_seq)
        masked_lengths = self.random_span_lengths(num_spans, masks_per_seq)

        source_ids = []
        source_labels = []

        targets = []

        # select random special tokens
        special_tokens = self.special_token_ids[torch.randperm(len(self.special_token_ids))[:num_spans]]

        if not self.args.infill_random_order:
            special_tokens = special_tokens.sort().values

        source_labels.append(seq[:1])

        for unmasked_length, masked_length, special_token in zip(unmasked_lengths, masked_lengths, special_tokens):
            source_ids.append(seq[:unmasked_length])
            source_ids.append(special_token[None])

            seg_labels = seq[1:unmasked_length+1].clone()
            seg_labels[:self.args.infill_ignore_run_in] = -100

            source_labels.append(seg_labels)
            source_labels.append(torch.tensor([-100]))
            seq = seq[unmasked_length:]

            targets.append((special_token, seq[:masked_length]))
            seq = seq[masked_length:]

        source_ids.append(seq)
        source_ids.append(torch.tensor([self.tokenizer.bos_token_id]))
        source_labels.append(seq[1:])
        source_labels.append(torch.tensor([-100]))

        targets.sort(key=lambda x: x[0])
        target_ids = []
        for target_token, target_seq in targets:
            target_ids.append(target_token[None])
            target_ids.append(target_seq)
        target_ids.append(torch.tensor([self.tokenizer.bos_token_id]))

        input_ids = torch.cat(source_ids + target_ids)
        labels = torch.cat(source_labels + target_ids)

        # print("N:", N, "masks_per_seq:", masks_per_seq, "num_spans:", num_spans, "mean_span_length:", mean_span_length, "infill_rate:", infill_rate)
        # if torch.distributed.get_rank() == 0:
        #     for i, (a, b) in enumerate(zip(input_ids.tolist(), labels.tolist())):
        #         print(i, a, b)

        return input_ids, labels

    @torch.no_grad()
    def __call__(self, features):
        input_ids_ = [torch.tensor(item["input_ids"], dtype=torch.long) for item in features]
        labels_ = [torch.tensor(item["input_ids"], dtype=torch.long) for item in features]

        for i in range(len(input_ids_)):
            if torch.rand(1) < self.args.infill_proportion:
                input_ids_[i], labels_[i] = self.span_masking(input_ids_[i])

        bsz = len(features)
        max_length = max(len(seq) for seq in input_ids_)

        input_ids = torch.full((bsz, max_length), self.pad_token_id, dtype=torch.long)

        attention_mask = torch.zeros(bsz, max_length, dtype=torch.long)
        labels = torch.full((bsz, max_length), -100, dtype=torch.long)

        for i, (seq, label_seq) in enumerate(zip(input_ids_, labels_)):
            input_ids[i, :len(seq)] = seq
            labels[i, :len(seq)] = label_seq
            attention_mask[i, :len(seq)] = 1

        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)

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
    if args.config_overrides_json:
        logger.info(f"Overriding config: {args.config_overrides_json}")
        config.update(json.loads(args.config_overrides_json))
        logger.info(f"New config: {config}")

    config.pad_token_id = 0

    half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
    if args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            torch_dtype=(half_dtype if args.lora or args.lora_path else None),
        )
    else:
        logger.warning(f"Initializing new LlamaForCausalLM from scratch")
        model = LlamaForCausalLM(config)

    if args.lora or args.lora_path:
        from peft import PeftModel, get_peft_model, LoraConfig, TaskType
        if args.lora_path:
            logger.info(f"Loading LoRA model from {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                modules_to_save=args.lora_modules_to_save,
            )
            model = get_peft_model(model, peft_config)

        for name, module in model.named_modules():
            if "lora" in name and sum(p.numel() for p in module.parameters()) > 0:
                module._fsdp_wrap = True
        model.print_trainable_parameters()

    if args.half_precision_training:
        model = model.to(half_dtype)

    logger.info(f"Model: {model}")

    # load_datasets
    if training_args.do_train:
        if len(args.tokenized_train_dataset) > 1:
            train_dataset = datasets.concatenate_datasets([
                datasets.load_from_disk(ds) for ds in args.tokenized_train_dataset
            ])
        else:
            train_dataset = datasets.load_from_disk(args.tokenized_train_dataset[0])
        logger.warning(f"train_dataset sequences: {len(train_dataset):,}")

        if args.sort_by is not None:
            logger.info(f"Sorting train_dataset by {args.sort_by} in {'reverse' if args.reverse_sort else 'normal'} order...")
            train_dataset = train_dataset.sort(args.sort_by, reverse=args.reverse_sort)
            logger.info(f"Sorting completed!")

    if training_args.do_eval:
        eval_dataset = datasets.load_from_disk(args.tokenized_validation_dataset)
        logger.warning(f"eval_dataset sequences: {len(eval_dataset):,}")
    if training_args.do_predict:
        test_dataset = datasets.load_from_disk(args.tokenized_test_dataset)
        logger.warning(f"test_dataset sequences: {len(test_dataset):,}")

    data_collator = DataCollator(args, training_args, tokenizer)
    if args.infill_proportion > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if trainer.is_fsdp_enabled:
        # Override accelerate defaults
        trainer.accelerator.state.fsdp_plugin.limit_all_gathers = True
        trainer.accelerator.state.fsdp_plugin.sync_module_states = False

        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
        trainer.accelerator.state.fsdp_plugin.backward_prefetch = BackwardPrefetch.BACKWARD_PRE

        # Identify which modules have "_fsdp_wrap" attribute set to True and wrap these
        def fsdp_policy_fn(module):
            return getattr(module, "_fsdp_wrap", False)

        # Identify which modules have "layer" in their class name and use these
        # as the basic FSDP blocks that are sharded and exchanged between GPUs
        # def layer_policy_fn(module):
            # return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=fsdp_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
        # trainer.accelerator.state.fsdp_plugin.use_orig_params = True

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

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

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(test_dataset=test_dataset)
        print(predictions)
        predictions = predictions.predictions
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        with open('dump.json', 'w') as f:
            print(json.dumps(predictions), file=f, flush=True)


if __name__ == "__main__":
    main()
