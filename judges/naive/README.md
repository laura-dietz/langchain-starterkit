# Naive Jude

A naive judge that just uses the length of responses respectively a random score as evaluation measure.

# Submit to TIRA

```
tira-cli code-submission \
            --dry-run \
            --path . \
            --file judges/naive/Dockerfile \
            --task trec-auto-judge \
            --dataset kiddie-20260605-training \
            --command 'auto-judge run --workflow /auto-judge/judges/naive/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```
