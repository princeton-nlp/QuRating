import os
import argparse
import multiprocessing
import time
import random
from math import ceil, floor
from functools import partial, reduce
from typing import List, Optional, Tuple, Iterable
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import torch
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset

def maybe_concatenate(datasets):
    return (concatenate_datasets(datasets) if len(datasets) >= 1 else datasets[0])


def selector(dataset_paths: List[str],
             selected_rows: Optional[npt.NDArray[np.float32]] = None,
             sequence_lengths: Optional[npt.NDArray[np.int32]] = None,
             seq_len_field: Optional[str] = None,
             json: bool = False) -> Iterable[Tuple[Dataset, int]]:
    rows_processed = 0

    for path in dataset_paths:
        print(f"Loading {path}")
        if json: # json datasets are not sharded
            try:
                dataset = load_dataset("json", data_files=[path], split="train")
            except Exception as e:
                print("Encountered exception")
                print(e)
                wait_time = random.random() * 10
                print("Waiting for {} seconds".format(wait_time))
                time.sleep(wait_time)
                dataset = load_dataset("json", data_files=[path], split="train", download_mode='force_redownload')
        else:
            dataset = load_from_disk(path)

        num_rows = len(dataset)

        if selected_rows is not None:
            subset_mask = (selected_rows >= rows_processed) & (selected_rows < rows_processed + num_rows)
            subset_selected_rows = selected_rows[subset_mask]
            subset_selected_rows.sort()

            dataset = dataset.select(subset_selected_rows - rows_processed)
        else:
            subset_mask = slice(rows_processed, rows_processed + num_rows)
        rows_processed += num_rows

        if sequence_lengths is not None:
            num_tokens = sequence_lengths[subset_mask].sum()
        else:
            num_tokens = sum(dataset[seq_len_field])

        yield dataset, num_tokens


def sharder(dataset_iterator: Iterable[Tuple[Dataset, int]], tokens_per_shard):
    tokens_in_shard = 0
    current_shard = []

    for dataset, tokens in dataset_iterator:
        tokens_in_shard += tokens
        current_shard.append(dataset)

        if tokens_in_shard >= tokens_per_shard:
            current_shard = maybe_concatenate(current_shard)
            num_rows = len(current_shard)

            # Assume that the number of tokens per row is roughly the same
            row_splits = [round(t / tokens_in_shard * num_rows) for t in range(0, tokens_in_shard, tokens_per_shard)]

            for a, b in zip(row_splits[:-1], row_splits[1:]):
                yield current_shard.select(range(a, b))
                tokens_in_shard -= tokens_per_shard

            current_shard = [current_shard.select(range(row_splits[-1], num_rows))]
            tokens_in_shard = round(len(current_shard[0]) / num_rows * tokens_in_shard)

    if len(current_shard) > 0 and tokens_in_shard > 0:
        yield maybe_concatenate(current_shard)


def load_attributes_for_dataset(path, metric_field, reference_field, seq_len_field, domain_field, seed=42):
    try:
        print(f"Loading {path}")
        dataset = load_from_disk(path)
        if metric_field is None:
            metrics = np.ones(len(dataset))
        else:
            metrics = sum(
                np.array(dataset[field], dtype=np.float32)
                for field in metric_field
            )

        if reference_field is not None:
            metrics -= np.array(dataset[reference_field], dtype=np.float32)

        if domain_field is not None:
            if "." in domain_field:
                domain_field, *dict_fields = domain_field.split(".")
                get_field = lambda x: reduce(dict.get, [x, *dict_fields])
            else:
                get_field = lambda x: x

            domains = np.array([hash(get_field(x)) % 2**32 for x in dataset[domain_field]], dtype=np.uint32)
        else:
            domains = None

        seq_len = np.array(dataset[seq_len_field], dtype=np.int32)

        return metrics, seq_len, domains
    except Exception as e:
        print("*****"* 10)
        print(f"PROBLEM WITH LOADING ATTRIBUTES FOR PATH '{path}'")
        print("*****" * 10)
        raise e


def load_domains_for_dataset(path):
    return torch.load(path).numpy()

def special_metrics_for_dataset(path):
    return np.load(path)

def load_attributes_for_all_datasets(args):
    if args.attributes:
        assert len(args.inputs) == len(args.attributes), f"{len(args.inputs)} != {len(args.domains)}"
    if args.domains:
        assert len(args.inputs) == len(args.domains), f"{len(args.inputs)} != {len(args.domains)}"
    with multiprocessing.Pool(args.num_workers) as pool:
        attributes = pool.map(
            partial(
                load_attributes_for_dataset,
                seq_len_field=args.seq_len_field,
                metric_field=args.metric_field,
                reference_field=args.reference_field,
                domain_field=args.domain_field,
                seed=args.seed),
            args.attributes or args.inputs)

        if args.domain_field is None and args.domains:
            domains = pool.map(load_domains_for_dataset, args.domains)
        else:
            domains = None

        if args.metric_field is None and args.metrics:
            metrics = pool.map(special_metrics_for_dataset, args.metrics)
        else:
            metrics = None

    # for i in range(len(metrics)):
    #     assert len(metrics[i]) == len(attributes[i][0]), f"{len(metrics[i])} != {len(attributes[i][0])}"

    if metrics is not None:
        metrics = np.concatenate(metrics)
    else:
        metrics = np.concatenate([m[0] for m in attributes])

    num_tokens = np.concatenate([m[1] for m in attributes])
    if args.domain_field is not None:
        domains = np.concatenate([m[2] for m in attributes])
    elif domains is not None:
        domains = np.concatenate(domains)
    else:
        domains = None

    assert len(metrics) == len(num_tokens)
    assert domains is None or len(domains) == len(num_tokens)

    return metrics, num_tokens, domains


def percentile_indices(metrics, num_tokens, total_num_tokens, tokens_to_select, margin):
    print(f"Sorting...")
    indices = np.argsort(metrics)
    # TODO replace with argpartition / topk of upper_limit followed by sorting

    if tokens_to_select == 0:
        return indices[:0], num_tokens[:0]

    upper_limit = ceil(len(metrics) * (tokens_to_select / total_num_tokens + margin))
    indices = indices[:upper_limit]

    selected_num_tokens = num_tokens[indices]
    cum_tokens = np.cumsum(selected_num_tokens)
    cutoff = np.argmax(cum_tokens >= tokens_to_select)

    if cum_tokens[cutoff] < tokens_to_select:
        print(f"Margin insufficient: {cum_tokens[cutoff]}/{tokens_to_select}")
        return percentile_indices(metrics, num_tokens, total_num_tokens, tokens_to_select, 2*margin)

    return indices[:cutoff + 1], selected_num_tokens[:cutoff + 1]


def equi_domain_percentile_indices(metrics, num_tokens, total_num_tokens, domains, tokens_to_select, margin):
    unique_domains = np.unique(domains)

    indices = []
    selected_num_tokens = []

    for domain in tqdm(unique_domains):
        domain_mask = (domains == domain)
        domain_metrics = metrics[domain_mask]
        domain_num_tokens = num_tokens[domain_mask]

        total_domain_num_tokens = np.sum(domain_num_tokens)
        tokens_to_select_in_domain = int(total_domain_num_tokens / total_num_tokens * tokens_to_select)
        print("Domain index:", domain, "Domain size:", len(domain_metrics), "Domain tokens:", total_domain_num_tokens, "Select:", tokens_to_select_in_domain)

        domain_indices, domain_num_tokens = percentile_indices(
            domain_metrics,
            domain_num_tokens,
            total_domain_num_tokens,
            tokens_to_select_in_domain,
            margin)
        indices.append(np.where(domain_mask)[0][domain_indices])
        selected_num_tokens.append(domain_num_tokens)

    return (
        np.concatenate(indices),
        np.concatenate(selected_num_tokens)
    )


def main(args):
    if args.tokens > 0:
        metrics, num_tokens, domains = load_attributes_for_all_datasets(args)

        np.random.seed(args.seed)
        if args.normalize:
            metrics = (metrics - metrics.mean()) / metrics.std()

        if args.temperature != 0.0:
            metrics = metrics / args.temperature

        if args.sample and args.temperature != 0.0:
            metrics += np.random.gumbel(size=len(metrics)) # Use topk-gumbel trick

        if args.select_bottom:
            metrics = metrics
        else:
            metrics = -metrics # We use argsort and always select the first indices


        print(f"Counting tokens...")
        total_num_tokens = np.sum(num_tokens)
        print(f"{total_num_tokens} tokens")

        if domains is None:
            indices, num_tokens = percentile_indices(metrics, num_tokens, total_num_tokens, args.tokens, args.margin)
        else:
            indices, num_tokens = equi_domain_percentile_indices(metrics, num_tokens, total_num_tokens, domains, args.tokens, args.margin)

        dataset_generator = selector(args.inputs, indices, num_tokens, json=args.json)
    else:
        dataset_generator = selector(args.inputs, seq_len_field=args.seq_len_field, json=args.json)

    for shard, dataset in enumerate(sharder(dataset_generator, args.tokens_per_shard)):
        print(f"Saving shard {shard}")
        dataset.save_to_disk(args.output + f"/{shard}", num_proc=args.num_workers)
    num_shards = shard + 1

    print("Renaming shards")
    for shard in range(num_shards):
        os.rename(args.output + f"/{shard}", args.output + f"/{shard}-{num_shards}{args.shard_suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for selecting percentile from a dataset.")
    parser.add_argument("inputs", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("output", type=str, help="Path to the output dataset.")

    parser.add_argument("--attributes", type=str, nargs="+", default=[], help="Path to attribute datasets, should match input size")
    parser.add_argument("--domains", type=str, nargs="+", default=[], help="Path to domains datasets, should match input size")
    parser.add_argument("--metrics", type=str, nargs="+", default=[], help="Path to metrics datasets, should match input size")
    parser.add_argument("--json", action="store_true", help="Save as json instead of arrow.")

    parser.add_argument("-n", "--seq_len_field", type=str, help="Num token field.", default="input_len")
    parser.add_argument("-m", "--metric_field", type=str, nargs="+", help="Field for metric. Leave empty for random selection", default=None)
    parser.add_argument("-r", "--reference_field", type=str, help="Field for reference. Leave empty for no reference", default=None)
    parser.add_argument("-d", "--domain_field", type=str, help="Domain field for equi-proprotional selection", default=None)

    parser.add_argument("-T", "--tokens", type=int, help="Tokens to select", default=5_000_000_000)

    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for logit sampling sampling')
    parser.add_argument("--sample", action="store_true", help="Use metrics as logits and sample without replacement")
    parser.add_argument("--normalize", action="store_true", help="Normalize metrics")
    parser.add_argument("--select_bottom", action="store_true", help="Select bottom scores.")

    parser.add_argument("--tokens_per_shard", type=int, help="Tokens per shard", default=500_000_000)
    parser.add_argument("--shard_suffix", type=str, help="Suffix for shard names", default="")

    parser.add_argument("--margin", type=float, default=0.1, help="Extra proportion for sampling enough data to deal with variable sequence lengths.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random selection. NOTE: seed will also depend on folder name of dataset.")
    parser.add_argument("-w", "--num_workers", type=int, default=None, help="Workers for saving.")

    # parser.add_argument("--segmentwise_metric", action="store_true", help="Use segmentwise metric")
    # parser.add_argument("--min_length", default=0, type=int, help="If document is post-processed.")
    # parser.add_argument("--max_length", default=999999999999999999, type=int, help="If document is post-processed.")

    args = parser.parse_args()
    print("Arguments:", args)
    main(args)
