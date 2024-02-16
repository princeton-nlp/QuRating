from datasets import load_from_disk, concatenate_datasets
import argparse
import os
from collections import defaultdict
import shutil

def transform(args):
    shared_columns = ["texts", "indices", "examples"]
    datasets = defaultdict(list)
    for path in args.inputs:
        prefix, suffix = path.split(args.common_delimiter)
        dataset = load_from_disk(path)
        for column in dataset.column_names:
            if column not in shared_columns:
                dataset = dataset.rename_column(column, suffix + "_" + column)
        datasets[prefix].append(dataset)

    for key, category_datasets in datasets.items():
        assert len(category_datasets) == 4 # 4 categories

        for shared_column in shared_columns:
            assert all(ds[shared_column] == category_datasets[0][shared_column] for ds in category_datasets[1:])
            category_datasets = [category_datasets[0]] + [ds.remove_columns(shared_column) for ds in category_datasets[1:]]

        dataset = concatenate_datasets(category_datasets, axis=1)
        if os.path.exists(key):
            raise ValueError(f"Path {key} already exists.")
        dataset.save_to_disk(key)

def cleanup(args):
    for path in args.inputs:
        prefix, _ = path.split(args.common_delimiter)
        if os.path.exists(prefix):
            shutil.rmtree(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for merging judgmement data across criteria.")
    parser.add_argument("inputs", type=str, nargs="+", help="Path to the judgments datasets")
    parser.add_argument("--common_delimiter", type=str, nargs="--", help="The delimiter to detect the name of the dataset")

    args = parser.parse_args()

    transform(args)
    cleanup(args)
