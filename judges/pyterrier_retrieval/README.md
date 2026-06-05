
```
tira-cli code-submission \
            --dry-run \
            --path . \
            --file judges/pyterrier_retrieval/Dockerfile \
            --task trec-auto-judge \
            --dataset kiddie-20260605-training \
            --mount-hf-model facebook/fasttext-en-vectors \
            --command 'auto-judge run --workflow /auto-judge/judges/pyterrier_retrieval/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```

