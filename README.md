# QuRating: Selecting High-Quality Data for Training Language Models
This is the official repository for our ICML'24 paper [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739)
and contains code for (1) collecting LLM quality judgments (2) training QuRater models (3) selecting and sampling data (4) training LMs (5) reproducing the analysis in the paper *(in-progress)*.

<br>
<p align="center">
<img src="assets/overview.png" width="600">
</p>
<br>

**Guidance on Responsible Use**

In the paper, we document various types of bias that are present in the quality ratings/QuRater model (biases related to domains, topics, social roles, regions and languages - see Section 6 of the paper).
Hence, be aware that data selection with QuRating could have unintended and harmful effects on the language model that is being trained. We strongly recommend a comprehensive evaluation of the language model for these and other types of bias, particularly before real-world deployment. We hope that releasing the data/models can facilitate future research aimed at uncovering and mitigating such biases. Note that the quality ratings do not measure the social or literary value of a text and should *not* be used for textual or demographic studies.

## Datasets
* We release the **_250K pairwise GPT-3.5-turbo judgments_** at [princeton-nlp/QuRating-GPT3.5-Judgments](https://huggingface.co/datasets/princeton-nlp/QuRating-GPT3.5-Judgments) to faciliate data inspection and the training of custom QuRater models.
  * We provide an additional 7140 pairwise GPT-3.5 judgments within 5 domains for test evaluation at [princeton-nlp/QuRating-GPT3.5-Judgments-Test](https://huggingface.co/datasets/princeton-nlp/QuRating-GPT3.5-Judgments-Test).
* ‼️  The **_260B token QuRatedPajama_** can be found and downloaded at [princeton-nlp/QuRatedPajama-260B](https://huggingface.co/datasets/princeton-nlp/QuRatedPajama-260B)
 * To explore the annotated QuRatedPajama, we release a **_1B token subset_** at [princeton-nlp/QuRatedPajama-1B_tokens_for_analysis](https://huggingface.co/datasets/princeton-nlp/QuRatedPajama-1B_tokens_for_analysis).
This dataset contains normalized quality scores for each criterion and topic cluster assignments for documents from C4 and CommonCrawl. It it used extensively throughout Section 6 of the paper.

## Models
**_‼️  The QuRater model fine-tuned from [ShearedLlama-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B) can be found at [princeton-nlp/QuRater-1.3B](https://huggingface.co/princeton-nlp/QuRater-1.3B) on HuggingFace hub ‼️_**


#### Language Models
We train 1.3B language models on 30B tokens selected from 260B tokens using different data selection methods. All 30 models from our experiments can be found on HuggingFace hub:

* Baselines
  * Uniform Sampling: [princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling)
  * DSIR
    * Target: Wikipedia (en): [princeton-nlp/lm-1.3B-select_30B_tokens_by-dsir_wikipedia_en](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-dsir_wikipedia_en)
    * Target: Book: [princeton-nlp/lm-1.3B-select_30B_tokens_by-dsir_book](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-dsir_book)
  * Perplexity filtering (based on ShearedLlama-2.7b)
    * keep lowest [princeton-nlp/lm-1.3B-select_30B_tokens_by-perplexity-bottom_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-perplexity-bottom_k)
    * keep highest [princeton-nlp/lm-1.3B-select_30B_tokens_by-perplexity-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-perplexity-top_k)
* Curriculum models - trained on the same dataset as the *Uniform Sampling* model, but we train on the samples in the order in which they are resampled using QuRating.
  * High-to-Low Required Expertise: <br> [princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling-curriculum-high_to_low-required_expertise](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling-curriculum-high_to_low-required_expertise)
  * Low-to-High Required Expertise: <br> [princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling-curriculum-low_to_high-required_expertise](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-uniform-sampling-curriculum-low_to_high-required_expertise)
* QuRating models:
  | Quality Criterion     | Top-k Selection  | Sample with Temperature 1.0  | Sample with Temperature 2.0 |
  |:-------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | Writing Style      | [princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-top_k)           | [princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-sample_with_temperature1.0)           | [princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-writing_style-sample_with_temperature2.0)           |
  | Facts & Trivia   | [princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-top_k)     | [princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-sample_with_temperature1.0)     | [princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-facts_and_trivia-sample_with_temperature2.0)     |
  | Educational Value  | [princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-top_k)   | [princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature1.0)   | [princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-educational_value-sample_with_temperature2.0)   |
  | Required Expertise | [princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-top_k) | [princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-sample_with_temperature1.0) | [princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-required_expertise-sample_with_temperature2.0) |
  | Mix of Criteria | - | - | [princeton-nlp/lm-1.3B-select_30B_tokens_by-mix_of_criteria-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-mix_of_criteria-sample_with_temperature2.0) |
  | Inverse Writing Style      | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-top_k)           | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-sample_with_temperature1.0)           | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_writing_style-sample_with_temperature2.0)           |
  | Inverse Facts & Trivia   | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-top_k)     | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-sample_with_temperature1.0)     | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_facts_and_trivia-sample_with_temperature2.0)     |
  | Inverse Educational Value  | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-top_k)   | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-sample_with_temperature1.0)   | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_educational_value-sample_with_temperature2.0)   |
  | Inverse Required Expertise | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-top_k](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-top_k) | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-sample_with_temperature1.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-sample_with_temperature1.0) | [princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-sample_with_temperature2.0](https://huggingface.co/princeton-nlp/lm-1.3B-select_30B_tokens_by-inverse_required_expertise-sample_with_temperature2.0) |
* Comparison: Uniform Sampling trained with 45B tokens: [princeton-nlp/lm-1.3B-select_45B_tokens_by-uniform-sampling](https://huggingface.co/princeton-nlp/lm-1.3B-select_45B_tokens_by-uniform-sampling)

## Experiments
### Installing the Repo
Clone this repo and setup a new environment based on `python 3.9`. Install the requirements in the following order:
```bash
pip install packaging==23.2
pip install torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1
pip install -r requirements.txt
```

### Collecting Judgment Data
The scripts `prompting/score_pairwise.py` and `prompting/score_individual.py` can be used to collect pairwise and individual judgments, respectively. Both scripts operate on huggingface datasets containing documents.
The folder `prompting/templates/` contains the templates used in the paper. Example usage:
```bash
python prompting/score_pairwise.py <input path> <output path> \
    -n 1000 -k 2 \
    --model gpt-3.5-turbo \
    --template_file prompting/templates/pairwise_educational_value.txt \
    --tokens_min 256 --tokens_max 512 --probability_tokens_max 0.5 \
    --text_field text
```
This selects the first 1000 documents in the input dataset and creates 500 adjacent pairs of documents, which will be compared using `gpt-3.5-turbo` (the different configurations can be found in `prompting/openai_util.py`). The output dataset will be stored as `<output path>`.

### QuRater Training
Run `TrainQuRater.sh` to train the QuRater models. You can override the default hyperparameters by setting environment variables (see `TrainQuRater.sh` for more details), e.g.:
```bash
BSZ=512 SEQ=4 ./TrainLM.sh
```
`SEQ` is the number of sequences per device in each forward pass and should as large as hardware allows.
The script automatically detects the number of GPUs and uses gradient accumulation to achieve a total batch size `BSZ`.
By default the script downloads the [ShearedLlama-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B) and [QuRating-GPT3.5-Judgments](https://huggingface.co/datasets/princeton-nlp/QuRating-GPT3.5-Judgments) dataset from huggingface hub.

## Annotating Data with Quality Ratings
[qurater_annotate.py](https://github.com/princeton-nlp/QuRating/blob/main/data_tools/qurater_annotate.py) takes a dataset and a QuRater model and adds new columns to the dataset for the quality ratings. Example usage for jsonl documents with a text column (`{"text": "..."}`):
```
python -m data_tools.qurater_annotate json <output path for annotated dataset> \
    -F <path to jsonl files> \
    -M princeton-nlp/QuRater-1.3B \
    --text_field text
    --labels writing_style required_expertise facts_and_trivia educational_value
```
The resulting dataset can be inspected via huggingface datasets via `datasets.load_from_disk(...)` and will contain additional columns like `writing_style_chunks` (segment-level quality ratings) and `writing_style_average` (document-level average) for each of the four criteria, similar to the extra columns in [QuRatedPajama-260B](https://huggingface.co/datasets/princeton-nlp/QuRatedPajama-260B).
The order of these labels in the argument list corresponds to their head index of the QuRater model.


## Selecting Data by Quality Ratings
[select_subset.py](https://github.com/princeton-nlp/QuRating/blob/main/data_tools/select_subset.py) loads input datasets, and perform top-k selection or sampling by treating a column as logits. It selects data until a token budget is reached and requires that the dataset contains a column with number of tokens per entry.
Here's an example use-case for selecting 1B tokens according to the `educational_value_average` field with temperature 2.0:

```
python -m data_tools.select_subset <path to annotated dataset> <output path for subset> \
    --metric_field educational_value_average \
    --seq_len_field <column name for sequence lengths> \
    --tokens 1_000_000_000 \
    --temperature 2.0 \
    --normalize \
    --num_workers 8
```
where `--normalize` normalizes the mean/std of the metric over the training set. If your data has a domain field, you can select a proportional number of examples from each domain by adding `--domain_field <column name for domain string>`.
This scripts writes multiple local HF datasets under the output path (useful for large datasets). They can all be read via
`datasets.concatenate_datasets([datasetes.load_from_disk(ds) for ds in sorted(glob.glob("<output path>/*"))])`

### Language Model Training
Run `TrainLM.sh` to train language models:
```
DATASET=<path to dataset> BSZ=2048 SEQ=8 ./TrainLM.sh
```
The dataset path is a local path to a huggingface dataset saved to disk.
The dataset needs to have a column, `input_ids`, containing sequences of tokenized text, chunked to the desired maximum sequence length.
We provide a script `data_tools/tokenize_dataset.py` to tokenize and chunk text datasets.
Note that the training script is only compatible with Llama architectures.
Please refer to `TrainLM.sh` to see more options for hyperparameters.

Curriculum training can be enabled by sorting the data according to another column of the dataset:
```
DATASET=<path to dataset> ./TrainLM.sh --ordered --sort_by <column name> [--reverse_sort]
```
where `--ordered` means that the data is trained on in the same order as found in the dataset rather than randomly shuffled.

### Citation
```bibtex
@inproceedings{wettig2024qurating,
   title={{QuRating}: Selecting High-Quality Data for Training Language Models},
   author={Wettig, Alexander and Gupta, Aatmik and Malik, Saumya and Chen, Danqi},
   booktitle={International Conference on Machine Learning (ICML)},
   year={2024}
}
```
