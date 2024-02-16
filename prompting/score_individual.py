from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import argparse
import numpy as np
from prompting.openai_util import query_openai


class Grader:
    @staticmethod
    def add_args(parser):
        parser.add_argument("-b", "--batch_size", type=int, default=2)
        parser.add_argument("-n", "--num_examples", type=int, default=100)
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--num_examples_proportion", type=float, default=None)
        parser.add_argument("--num_examples_proportion_start", type=float, default=0.0)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4", "claude-2"])

        parser.add_argument("-g", "--generations", type=int, default=20)
        parser.add_argument("--template", type=str, default="")
        parser.add_argument("--template_file", type=str, default="annotate/templates/individual_default.txt")
        parser.add_argument("--grade_range", type=int, nargs=2, default=[1, 10])

        parser.add_argument("--text_field", type=str, default="text")
        parser.add_argument("--token_field", type=str, default="input_ids")
        parser.add_argument("--tokens_min", type=int, default=256)
        parser.add_argument("--tokens_max", type=int, default=512)
        parser.add_argument("--probability_tokens_max", type=float, default=0.5)
        parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")

        parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")

    def __init__(self, args):
        self.args = args
        if not self.args.template:
            if self.args.template_file:
                with open(self.args.template_file) as f:
                    self.args.template = f.read()
            else:
                raise ValueError("Either --template or --template_file must be specified.")

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    def extract_excerpt(self, text, index, num_tokens, token_ids=None):
        if token_ids is None:
            # heuristic for faster tokenization
            max_character_length = self.args.tokens_max * 40
            if len(text) > max_character_length:
                np.random.seed(self.args.seed + index + self.offset + 1)
                start_pos = np.random.randint(0, len(text) - max_character_length + 1)
                text = text[start_pos:start_pos + max_character_length]

            token_ids = self.tokenizer(text, truncation=False, padding=False, add_special_tokens=False).input_ids

        if len(token_ids) <= self.args.tokens_max:
            return text

        np.random.seed(self.args.seed + index + self.offset)
        start_pos = np.random.randint(0, len(token_ids) - self.args.tokens_max + 1)
        token_ids = token_ids[start_pos:start_pos + num_tokens]
        return self.tokenizer.decode(token_ids)

    def sample_num_tokens(self, indices):
        np.random.seed(self.args.seed + sum(indices) + self.offset)
        # use length tokens_max with probability probability_tokens_max, otherwise sample uniformly from [tokens_min, tokens_max]
        if self.args.probability_tokens_max > 0 and np.random.rand() < self.args.probability_tokens_max:
            return self.args.tokens_max
        else:
            return np.random.randint(self.args.tokens_min, self.args.tokens_max + 1)

    def parse_generations(self, generations):
        for generation in generations:
            try:
                yield float(generation)
            except:
                continue

    def __call__(self, examples, indices):
        num_tokens = self.sample_num_tokens(indices)

        if self.args.token_field in examples:
            texts = [self.extract_excerpt(text, index, num_tokens, token_ids) for text, token_ids, index in zip(examples[self.args.text_field], examples[self.args.token_field], indices)]
        else:
            texts = [self.extract_excerpt(text, index, num_tokens) for text, index in zip(examples[self.args.text_field], indices)]

        all_scores = []
        average_scores = []
        for text in texts:
            prompt = self.args.template.format(
                text=text,
                grade_min=self.args.grade_range[0],
                grade_max=self.args.grade_range[1]
            )

            generations = query_openai(
                prompt,
                self.args.model,
                system_prompt=self.args.system_prompt,
                generations=self.args.generations,
                labels=[str(label) for label in range(self.args.grade_range[0], self.args.grade_range[1] + 1)]
            )

            scores = np.array(list(self.parse_generations(generations)))
            all_scores.append(scores.tolist())
            if len(scores) == 0:
                average_score = -100
            else:
                average_score = np.mean(scores)
            average_scores.append(average_score)

        return {
            "index": indices,
            "text": texts,
            "predictions": all_scores,
            "average_prediction": average_scores,
        }

    def apply(self, dataset):
        if self.args.num_examples_proportion is not None:
            self.offset = int(len(dataset) * self.args.num_examples_proportion_start)
            self.num_examples = int(len(dataset) * self.args.num_examples_proportion)
        else:
            self.offset = self.args.offset
            self.num_examples = self.args.num_examples

        print("Example offset", self.offset)

        return (
            dataset
                .select(range(self.offset, self.offset + self.num_examples))
                .map(self.__call__,
                     with_indices=True,
                     remove_columns=dataset.column_names,
                     batched=True,
                     batch_size=self.args.batch_size,
                     keep_in_memory=True)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)

    parser.add_argument("--json", action="store_true")

    Grader.add_args(parser)

    args = parser.parse_args()
    print(args)

    print("Total number of examples:", args.num_examples)

    if args.json:
        dataset = load_dataset("json", data_files=[args.input], split="train", download_mode='force_redownload')
    else:
        dataset = load_from_disk(args.input)

    dataset = Grader(args).apply(dataset)

    print(f"Saving to {args.output}")
    dataset.save_to_disk(args.output)
