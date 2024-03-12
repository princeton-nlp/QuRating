import argparse
from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import time
from itertools import islice
import numpy as np
from tqdm import tqdm

def chunk(iterable, chunk_size, overlap=0):
    it = iter(iterable)
    cache = tuple(islice(it,overlap))
    stride = chunk_size - overlap
    while batch := tuple(islice(it, stride)):
        yield cache + batch
        cache = batch[stride - overlap:]


def chunk_with_position(iterable, chunk_size, overlap=0):
    position = 0
    for seg in chunk(iterable, chunk_size, overlap):
        yield seg, position
        position += len(seg) - overlap


def min_char_split(input_string, split_string, min_char, include_separator):
    if not split_string:
        yield input_string
        return

    start = 0

    while True:
        # Find the next occurrence of the split string
        index = input_string.find(split_string, start + min_char)

        # Break the loop if no more split strings are found
        if index == -1:
            yield input_string[start:]
            return

        # Add the substring to the results
        if include_separator:
            text = input_string[start:index + len(split_string)]
        else:
            text = input_string[start:index]

        if text:
            yield text
        start = index + len(split_string)


class TokenizeMapper:
    def __init__(self, args):
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=args.use_fast)

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    def random_chunk(self, seq, index):
        np.random.seed(index + self.args.seed)
        document_position = np.random.randint(0, max(len(seq) - self.args.max_length + 1, 1))
        seq_chunk = seq[document_position:document_position + self.args.max_length]
        yield seq_chunk, document_position

    def __call__(self, examples, indices=None):
        """Tokenizes the text from the input_field in the dataset.
        If the tokenized text is longer than max_length, splits it into chunks."""

        if self.args.input_tokens_field in examples:
            ids = examples[self.args.input_tokens_field]
        else:
            texts = examples[self.args.input_field]

            if self.args.min_num_chars_for_split >= 0 and any(len(text) > self.args.min_num_chars_for_split for text in texts):
                ids = []
                for text in texts:
                    if len(text) <= self.args.min_num_chars_for_split:
                        chunks = [text]
                    else:
                        chunks = min_char_split(text, self.args.min_num_chars_split_separator, self.args.min_num_chars_for_split, self.args.min_num_chars_include_separator)


                    ids.append([
                        token
                        for chunk in chunks
                        for token in self.tokenizer(chunk, truncation=False, add_special_tokens=False).input_ids
                    ])
            else:
                ids = self.tokenizer(texts, truncation=False, add_special_tokens=False).input_ids

        output = {}
        if not self.args.remove_columns:
            output.update({field: [] for field in examples})

        additional_fields = {self.args.input_field,
                             self.args.tokens_field,
                             self.args.length_field,
                             "document_position",
                             "document_index"}
        output.update({field: [] for field in additional_fields})

        for i, seq in enumerate(ids):
            chunk_generator = (
                self.random_chunk(seq, indices[i])
                if self.args.random_chunk else
                chunk_with_position(seq, self.args.max_length, self.args.overlap)
            )

            for seq_chunk, document_position in chunk_generator:
                if len(seq_chunk) < self.args.min_length:
                    break

                output[self.args.tokens_field].append(seq_chunk)
                output[self.args.length_field].append(len(seq_chunk))
                output["document_position"].append(document_position)
                output["document_index"].append(indices[i])

                if not self.args.remove_columns:
                    output[self.args.input_field].append(self.tokenizer.decode(seq_chunk))
                    for field in examples:
                        if field in additional_fields:
                            continue
                        output[field].append(examples[field][i])

        return output


def main(args):
    """Main function to perform tokenization."""

    print(f"Loading '{args.input}'...")
    if args.json:
        dataset = concatenate_datasets([
            load_dataset("json", data_files=path, split="train") for path in tqdm(args.input)
        ])
    else:
        dataset = concatenate_datasets([
            load_from_disk(path)
            for path in tqdm(args.input)
        ])
    print(f"Loaded '{args.input}'")

    dataset = dataset.shard(args.shard[1], args.shard[0], contiguous=True)

    if isinstance(dataset, dict):
        column_names = next(iter(dataset.values())).column_names
    else:
        column_names = dataset.column_names

    start_time = time.time()
    tokenized_dataset = dataset.map(
        TokenizeMapper(args),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=column_names,
        num_proc=args.num_workers,
        with_indices=True
    )
    print(f"Finished mapping in {time.time() - start_time:.2f}s")

    print(f"Saving to '{args.output}'...")
    tokenized_dataset.save_to_disk(args.output, num_proc=args.num_workers)
    print(f"Saved to '{args.output}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for tokenizing a dataset.")
    parser.add_argument("input", type=str, nargs="+", help="Path to the input datasets.")
    parser.add_argument("output", type=str, help="Path to the output tokenized dataset.")

    parser.add_argument("-t", "--tokenizer", type=str, default="gpt2", help="Tokenizer.")
    parser.add_argument("-F", "--use_fast", action="store_true", help="Use fast tokenizer.")

    parser.add_argument("-f", "--input_field", type=str, default="text", help="Field in the dataset to tokenize.")
    parser.add_argument("--input_tokens_field", type=str, default="input_ids", help="Maybe data is already tokenized?")
    parser.add_argument("--tokens_field", type=str, default="input_ids", help="Store tokenized data in this field")
    parser.add_argument("--length_field", type=str, default="length", help="Store number of tokens in this field")

    parser.add_argument("-L", "--max_length", type=int, default=2048, help="Maximum length of tokenized chunks.")
    parser.add_argument("-l", "--min_length", type=int, default=128, help="Minimum length of tokenized chunks.")

    parser.add_argument("-w", "--num_workers", type=int, default=None, help="Workers.")
    parser.add_argument("-r", "--remove_columns", action="store_true", help="Remove columns.")

    parser.add_argument("-b", "--batch_size", type=int, default=2048, help="Batch size for the map function.")
    parser.add_argument("-s", "--shard", type=int, nargs=2, default=[0, 1], help="Pick a shard of the dataset")
    parser.add_argument("-o", "--overlap", type=int, default=1, help="Overlap between contexts.")

    parser.add_argument("--random_chunk", default=False, action="store_true", help="One segment per document.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--json", action="store_true", help="Input is json dataset.")

    parser.add_argument("--min_num_chars_for_split", type=int, help="Split input string if above this length", default=-1)
    parser.add_argument("--min_num_chars_split_separator", type=str, help="Split separator", default=" ")
    parser.add_argument("--min_num_chars_include_separator", action="store_true", help="Include split separator", default=False)

    args = parser.parse_args()
    main(args)
