from typing import List
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import openai
import tiktoken
import os
import time
import json
from filelock import FileLock
import random

MODELS = {
    "gpt-3.5-turbo-16k": {
        "api_base": "https://pnlpopenai2.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2023-05-15",
        "deployment_name": "gpt-35-turbo-0613",
        "enc": tiktoken.encoding_for_model("gpt-3.5-turbo"),
        "prompt_cost_per_token": 0.003 / 1000,
        "response_cost_per_token": 0.004 / 1000,
    },
    "gpt-3.5-turbo": {
        "api_base": "https://pnlpopenai2.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2023-05-15",
        "deployment_name": "gpt-35-turbo-0613",
        "enc": tiktoken.encoding_for_model("gpt-3.5-turbo"),
        "prompt_cost_per_token": 0.0015 / 1000,
        "response_cost_per_token": 0.002 / 1000,
    },
    "gpt-4": {
        "api_base": "https://pnlpopenai3.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2023-05-15",
        "deployment_name": "gpt-4",
        "enc": tiktoken.encoding_for_model("gpt-4"),
        "prompt_cost_per_token": 0.03 / 1000,
        "response_cost_per_token": 0.06 / 1000,
    },
}

RANDOM = random.Random()


def query_openai(prompt: str,
                 model: str,
                 labels: List[str],
                 system_prompt: str = None,
                 generations: int = 1,
                 retries: int = 1,
                 log_file_path: int = "openai_api_cost.jsonl") -> List[str]:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = MODELS[model]["api_base"]
    openai.api_type = MODELS[model]["api_type"]
    openai.api_version = MODELS[model]["api_version"]
    deployment_name=MODELS[model]["deployment_name"]
    enc = MODELS[model]["enc"]

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
            response = openai.ChatCompletion.create(
                engine=deployment_name, model=deployment_name,
                messages=messages,
                temperature=1.0,
                max_tokens=max_tokens,
                n=generations,
                logit_bias=logit_bias,
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

    generations = [choice['message']['content'] for choice in response['choices']]

    usage = response["usage"]
    usage["prompt_cost"] = MODELS[model]["prompt_cost_per_token"] * usage["prompt_tokens"]
    usage["response_cost"] = MODELS[model]["response_cost_per_token"] * usage["completion_tokens"]
    usage["cost"] = usage["prompt_cost"] + usage["response_cost"]
    usage["model"] = deployment_name

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
