# TinyJudge

This is an minimal example for an TREC AutoJudge system that is intended to highlight the code submission process with a minimal example. The AutoJudge can be exeuted via the `auto-judge run` cli.

The tinyjudge system has two properties that are important for scaled execution:

- It reads the environment variables OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL to access the LLM.
- It uses a `CACHE_DIR` environment variable that configures where the client-side cache lives, this is important to ensure that the execution can resume in case of failures without involving high additional API costs and also for replicabiltiy.

# Execution of TinyJudge from Cached LLM Responses

We do it in two steps: first, execution from cache, second, evaluation.

We download an prompt cache of the execution of the TinyJudge on the kiddie dataset from tira and unzip it

```
wget https://www.tira.io/task/trec-auto-judge/user/webis/dataset/kiddie-20260605-training/download/2026-06-23-15-44-28.zip
unzip -j 2026-06-23-15-44-28.zip '2026-06-23-15-44-28/CACHE_DIR/*' -d example-cache-kiddie
```

Install dependencies
```
pip3 install autojudge[minimallm]
```

Now, we can set environment variables (we do not need an valid api key and base url as we use the previously downloaded cache.
For this, we first export the required environment variables:
```
export OPENAI_API_KEY=empty
export OPENAI_BASE_URL=empty
export CACHE_DIR=example-cache-kiddie
export OPENAI_MODEL=llama-3.1-8b-instant
```

Now, we can run the auto-judge using those environment variables and the downloaded cache:

```
auto-judge run \
    --workflow judges/tinyjudge/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir results/tinyjudge-kiddie
```

Now, we can evaluate the results:

```
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures \
    --truth-header \
    --eval-format ir_measures \
    --on-missing default \
    --input results/tinyjudge-kiddie/tinyjudge.eval.txt
```


The output should contain something like:

```
         Judge TruthMeasure             EvalMeasure  kendall  pearson  spearman  tauap_b  kendall@10
tinyjudge.eval    RELEVANCE FIRST_SENTENCE_RELEVANT 0.547723 0.702559  0.632456 0.416667    0.547723
```

# Submission to TIRA

When the above works, we are ready to submit to tira, the following needs to be done.


1. Ensure environment variables are exported:

```
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

Submit to tira:

```
tira-cli code-submission \
    --dry-run \
    --path . \
    --cache-behaviour deterministic \
    --mount-cache '$CACHE_DIR=EMPTY_DIR' \
    --task trec-auto-judge \
    --dataset kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --command 'auto-judge run --workflow /auto-judge/judges/tinyjudge/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

If everything worked, the output should look like this:

<img width="1807" height="259" alt="Screenshot_20260703_165606" src="https://github.com/user-attachments/assets/409ce157-53b9-48ee-bd36-a01d46f9d352" />

For explanation, the `--cache-behaviour deterministic` flag indicated that the software is intended to produce the same deterministic output when running from the cache, during the submission process, we run the software then twice and store that this is expected in the metadata, this helps us to incorporat the submitted software into replicability experiments.

For more details on the submission workflow and on how to prepare your software, please have a short look at the [TIRA participant documentation](https://docs.tira.io/participants/participate.html#prepare-your-submission). (The AutoJudge starter kit is already developed so that everything should be compatible with TIRA without much effort.)


## Run a published Naive Judge

This tiny judge is already published on TIRA and can also be executed locally via `tira-cli`:

```bash
OPENAI_API_KEY=empty OPENAI_BASE_URL=empty OPENAI_MODEL=llama-3.1-8b-instant \
tira-cli run local --approach trec-auto-judge/webis/tinyjudge \
    --input kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --mount-cache "CACHE_DIR=example-cache-kiddie"
```

The output should look like:

<img width="957" height="389" alt="Screenshot_20260703_170402" src="https://github.com/user-attachments/assets/1bcf0fb9-dda4-4819-9d6d-25e52c7fd62b" />

We can also use LLM with an empty cache directory, for this, first export environment variables:

```
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

then, we can run xy:

```bash
tira-cli run local \
    --approach trec-auto-judge/webis/tinyjudge \
    --input kiddie-20260605-training \
    --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
    --mount-cache "CACHE_DIR=EMPTY_DIR"
```

The output should look like:

<img width="957" height="389" alt="Screenshot_20260703_170630" src="https://github.com/user-attachments/assets/124ab00c-6312-4fbc-ac9a-b3a37fde2257" />

Please note that the result directory also contains the `CACHE_DIR` that could potentially be re-used. This is similar how we then run private submitted AutoJudges on the test datasets, potentially via different LLMs.)
