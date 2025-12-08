from typing import List
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import openai
import tiktoken
import os
import time
import json
from filelock import FileLock
import random


# Costs on Azure as of 2024-06-01. No liability for correctness.
MODEL_COSTS = {
    "gpt-3.5-turbo-0613": {
        "prompt_cost_per_token": 0.0015 / 1000,
        "response_cost_per_token": 0.002 / 1000,
    },
    "gpt-3.5-turbo-1106": {
        "prompt_cost_per_token": 0.001 / 1000,
        "response_cost_per_token": 0.002 / 1000,
    },
    "gpt-4-1106-preview": {
        "prompt_cost_per_token": 0.03 / 1000,
        "response_cost_per_token": 0.06 / 1000,
    },
    "gpt-4-0125-preview": {
        "prompt_cost_per_token": 0.03 / 1000,
        "response_cost_per_token": 0.06 / 1000,
    },
    "gpt-4-turbo-2024-04-09": {
        "prompt_cost_per_token": 0.01 / 1000,
        "response_cost_per_token": 0.03 / 1000,
    },
    "gpt-4o-2024-05-13": {
        "prompt_cost_per_token": 0.005 / 1000,
        "response_cost_per_token": 0.015 / 1000,
    },
}

# Add Azure shorthands
for model in list(MODEL_COSTS.keys()):
    MODEL_COSTS[model.replace("3.5", "35")] = MODEL_COSTS[model]

# Use a separate random for retries to avoid affecting the global state
RANDOM = random.Random()

client=openai.AzureOpenAI() # Replace with openai.Client() for OpenAI API

def query_openai(prompt: str,
                 model: str,
                 labels: List[str],
                 system_prompt: str = None,
                 generations: int = 1,
                 retries: int = 1,
                 log_file_path: int = "openai_api_cost.jsonl",
                 seed: int = 42) -> List[str]:
    enc = tiktoken.encoding_for_model(model)

    is_ok = False
    retry_count = 0

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    label_tokens = [enc.encode(label) for label in labels]
    logit_bias = {
        str(token): 100
        for token in set.union(*(set(tokens) for tokens in label_tokens))
    }
    max_tokens = max(len(tokens) for tokens in label_tokens)

    while not is_ok:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_tokens=max_tokens,
                n=generations,
                logit_bias=logit_bias,
                seed=seed,
            )
            is_ok = True
        except Exception as error:
            if "Please retry after" in str(error):
                timeout = int(str(error).split("Please retry after ")[1].split(" second")[0]) + 5*RANDOM.random()
                print(f"Wait {timeout}s before OpenAI API retry ({error})")
                time.sleep(timeout)
            elif retry_count < retries:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(2)
                retry_count += 1
            else:
                print(f"OpenAI API failed for {retry_count} times ({error})")
                return []

    generations = [choice.message.content for choice in response.choices]

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "model": model,
    }
    if model in MODEL_COSTS:
        usage["prompt_cost"] = MODEL_COSTS[model]["prompt_cost_per_token"] * response.usage.prompt_tokens
        usage["response_cost"] = MODEL_COSTS[model]["response_cost_per_token"] * response.usage.completion_tokens
        usage["cost"] = usage["prompt_cost"] + usage["response_cost"]

    with FileLock(log_file_path + ".lock"):
        with open(log_file_path, "a") as f:
            f.write(json.dumps(usage) + "\n")

    return generations


def query_anthropic(prompt: str,
                    model: str = "claude-2") -> List[str]:
    is_ok = False
    retry_count = 0

    while not is_ok:
        retry_count += 1
        try:
            anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            completion = anthropic.completions.create(
                model=model,
                max_tokens_to_sample=5,
                prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            )
            return [completion.completion]
        except Exception as error:
             if retry_count <= 2:
                 print(f"OpenAI API retry for {retry_count} times ({error})")
                 time.sleep(2)
             else:
                 print(f"OpenAI API failed for {retry_count} times ({error})")
                 return []
