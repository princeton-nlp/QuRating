import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets
from tqdm import tqdm
import os

def load_datasets(args):
    for path in tqdm(args.input, desc="Loading datasets"):
        if args.json:
            dataset = load_dataset("json", data_files=path, split="train")
        else:
            dataset = load_from_disk(path)
        yield (args.output + "/" + os.path.basename(path)), dataset

def concatenated_datasets(args):
    datasets = dict(load_datasets(args))
    yield args.output, concatenate_datasets(list(datasets.values()))

def main(args):
    if args.single:
        datasets = concatenated_datasets(args)
    else:
        datasets = load_datasets(args)

    for output_path, dataset in datasets:
        column_names = dataset.column_names

        columns_to_remove = set(column_names) - set(args.columns)
        dataset = dataset.remove_columns(columns_to_remove)

        print(f"Saving to '{output_path}'...")
        if args.parquet:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dataset.to_parquet(output_path, compression=args.compression, compression_level=args.compression_level)
        else:
            dataset.save_to_disk(output_path, num_proc=28)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for tokenizing a dataset.")
    parser.add_argument("input", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("output", type=str, help="Path to the output tokenized dataset.")
    parser.add_argument("--columns", type=str, nargs="+", help="Columns to keep.", default=["input_ids", "length"])
    parser.add_argument("--json", action="store_true", help="Input is json dataset.")
    parser.add_argument("-S", "--single", action="store_true", help="Concatenate into single input/output dataset.")
    parser.add_argument("--parquet", action="store_true", help="Store as parquet")

    parser.add_argument("--compression", type=str, default="zstd")
    parser.add_argument("--compression_level", type=int, default=3)

    args = parser.parse_args()
    main(args)
