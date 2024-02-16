from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_from_disk,concatenate_datasets
import torch
import torch.nn.functional as F
import math
import argparse
import os

from modeling.modeling_flash_llama import LlamaForCausalLM

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help="input dataset")
    parser.add_argument('output', type=str, help="output dataset")
    parser.add_argument("-m", "--model",
                        type=str,default='EleutherAI/pythia-160m-deduped',
                        help="Small model")
    parser.add_argument("-b", '--batch_size', type=int, default=32)
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use fp16 mixed precision training.')
    parser.add_argument('--bf16', action='store_true',
                        help='Whether to use bf16 mixed precision training.')
    parser.add_argument("-t", '--max_tokens', type=int, default=2048,
                        help='Maximum number of tokens the model can handle')
    parser.add_argument('--text_field', type=str, default='text',
                        help='Text field name')
    parser.add_argument("-f", '--field', type=str, default='avg_loglikelihood',
                        help='Field name')
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])

    args = parser.parse_args()

    return args


class Processor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None))
        self.model.eval()
        self.model.to(self.device)

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    @torch.inference_mode()
    def log_likelihood(self, sents):
        if "input_ids" in sents:
            seqs = sents["input_ids"]
        else:
            seqs = self.tokenizer(sents[self.args.text_field])

        max_length = min(max(len(x) for x in seqs), self.args.max_tokens+1)
        bsz = len(seqs)

        input_ids = torch.zeros(bsz, max_length, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_length, dtype=torch.long)

        for i, x in enumerate(seqs):
            input_ids[i,:len(x)] = torch.tensor(x[:max_length], dtype=torch.long)
            attention_mask[i,:len(x)] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        targets = input_ids[:,1:]
        input_ids = input_ids[:,:-1]
        target_attention_mask = attention_mask[:,1:]

        logprobs = self.model(input_ids).logits.float().log_softmax(dim=-1)

        loglikelihood = torch.gather(logprobs, -1, targets.unsqueeze(-1)).squeeze(-1)
        loglikelihood = loglikelihood * target_attention_mask.float()
        avg_loglikelihood = torch.sum(loglikelihood, dim=-1) / torch.sum(target_attention_mask, dim=-1).clamp(min=1)

        return avg_loglikelihood.cpu().tolist()

    def __call__(self, items):
        output = {}

        avg_loglikelihood = self.log_likelihood(items)
        output[self.args.field] = avg_loglikelihood
        return output

def main():
    args = init_args()
    dataset = load_from_disk(args.input)

    dataset = dataset.shard(args.shard[1], args.shard[0], contiguous=True)
    print(args)

    dataset = dataset.map(Processor(args), batched=True, batch_size=args.batch_size, keep_in_memory=True)

    dataset.save_to_disk(args.output)

if __name__ == "__main__":
    main()
