from glob import glob
from datasets import load_from_disk, load_dataset, concatenate_datasets
from tqdm import tqdm
import random

import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("filter", type=str, default="*")

args = parser.parse_args()

files = [
    sorted(glob(f"datasets/slimpajama-1000_4000-seg1024-equidomain-writing_style-t2.0bottom30B/{args.filter}-llama")),
    sorted(glob(f"datasets/slimpajama-1000_4000-seg1024-equidomain-facts_and_trivia-t2.0bottom30B/{args.filter}-llama")),
    sorted(glob(f"datasets/slimpajama-1000_4000-seg1024-equidomain-educational_value-t2.0bottom30B/{args.filter}-llama")),
    sorted(glob(f"datasets/slimpajama-1000_4000-seg1024-equidomain-required_expertise-t2.0bottom30B/{args.filter}-llama")),
]

transposed_files = list(zip(*files))

random.seed(42)

def mix_subsets(one_shard):
    ds = load_from_disk(one_shard[0])
    N = len(ds) # We always use the first dataset to determine how many examples this shard should have
    print("Initial dataset size:", N)

    split_points = [int(i*N/4) for i in range(5)]

    print(one_shard)

    counter = 0
    indices = []
    all_datasets = []
    for i, path in enumerate(tqdm(one_shard)):
        ds = load_from_disk(path)

        start_split_point = split_points[i]
        last_split_point = split_points[i+1]
        if last_split_point > len(ds):
            start_split_point -= (last_split_point - len(ds))
            last_split_point -= (last_split_point - len(ds))

        indices.extend(range(counter + start_split_point, counter + last_split_point))
        all_datasets.append(ds)
        counter += len(ds)

    all_datasets = concatenate_datasets(all_datasets)

    remaining_indices = list(set(range(N)) - set(indices))
    remaining_data = all_datasets.select(remaining_indices)
    ds = all_datasets.select(indices)

    duplicates = N - len(set(ds["text"]))

    while duplicates > 0:
        print(f"Found {duplicates} duplicates.")
        # Fill up any duplicates with random data points of the remaining data
        # Sample a bit more than we need to make sure we get enough unique data points
        indices = random.sample(range(len(remaining_data)), min(duplicates*10, len(remaining_data)))
        data_points = remaining_data.select(indices)

        ds = concatenate_datasets([ds, data_points])
        duplicates = N - len(set(ds["text"]))

    texts = np.array(ds["text"])
    _, indices = np.unique(texts, return_index=True)

    print("Dataset size pre dedup:", len(ds))
    ds = ds.select(indices)
    if len(ds) > N:
        ds = ds.select(range(N))
    assert len(ds) == N

    print("Dataset size post dedup:", len(ds))

    ds.save_to_disk(f"datasets/slimpajama-1000_4000-seg1024-equidomain-combined2-t2.0bottom30B/{one_shard[0].split('/')[-1]}", num_proc=28)

for one_shard in tqdm(transposed_files):
    mix_subsets(one_shard)

# with Pool(6) as p:
    # p.map(mix_subsets, transposed_files)