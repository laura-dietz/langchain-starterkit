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

Run it against the cache, i.e., we use only dummy values:
```
PYTHONPATH=../.. \
OPENAI_API_KEY=empty \
OPENAI_BASE_URL=empty \
CACHE_DIR=example-cache-kiddie \
OPENAI_MODEL=llama-3.1-8b-instant \
    auto-judge run --workflow workflow.yml --rag-responses ../../data/kiddie/runs/repgen/ --rag-topics ../../data/kiddie/topics/kiddie-topics.jsonl --out-dir my-results
```

Now, we can evaluate the results:

```
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard ../../data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures \
    --eval-format ir_measures \
    --input my-results/tinyjudge.eval.txt
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

Explanation:

xy.

# Re-Execution of approaches from TIRA

Approaches that have been published can be easily 


Re-Executing things with an cache:

```
OPENAI_API_KEY=empty OPENAI_BASE_URL=empty OPENAI_MODEL=llama-3.1-8b-instant \
    tira-cli run local --approach trec-auto-judge/webis/tinyjudge \
        --input kiddie-20260605-training \
        --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
        --mount-cache "CACHE_DIR=../../data/example-caches/2026-06-30-15-02-13/CACHE_DIR/"
```

vs

```
OPENAI_API_KEY=empty OPENAI_BASE_URL=empty OPENAI_MODEL=llama-3.1-8b-instant \
    tira-cli run local --approach trec-auto-judge/webis/tinyjudge \
        --input kiddie-20260605-training \
        --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
        --mount-cache "CACHE_DIR=../../data/example-caches/2026-07-01-05-48-25/CACHE_DIR/"
```
