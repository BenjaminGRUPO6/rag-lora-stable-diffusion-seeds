# Final System Evaluation

Generated at UTC: 2026-07-13T09:42:47.692584+00:00

## Scope

This report consolidates existing results and demo pipeline timings. No training was executed.
The `spotted` label is a visual category and is not treated as a fungal diagnosis.

## Vision

- Accuracy: 0.670498
- Macro precision: 0.741193
- Macro recall: 0.650534
- Macro-F1: 0.625955

### F1 by Class

- intact: 0.296296
- spotted: 0.724409
- immature: 0.688406
- broken: 0.641975
- skin_damaged: 0.778689

## RAG

- Hit@1: 0.450000
- Hit@3: 0.700000
- Hit@5: 0.800000
- MRR: 0.589167
- Retrieval latency ms: 12.807400
- Human review: {"metrics": "pending", "pending_queries": 20, "reviewed_queries": 0, "status": "pending"}

## LoRA

- Status: PARTIAL
- Confirmed parameters: {"base_model": {"source": "configs/lora_sd15.yaml", "value": "stable-diffusion-v1-5/stable-diffusion-v1-5"}, "gradient_accumulation_steps": {"source": "configs/lora_sd15.yaml", "value": 4}, "learning_rate": {"source": "configs/lora_sd15.yaml", "value": 0.0001}, "max_train_steps_full": {"source": "configs/lora_sd15.yaml", "value": 800}, "max_train_steps_initial": {"source": "configs/lora_sd15.yaml", "value": 100}, "mixed_precision": {"source": "configs/lora_sd15.yaml", "value": "fp16"}, "rank": {"source": "configs/lora_sd15.yaml", "value": 8}, "resolution": {"source": "configs/lora_sd15.yaml", "value": 512}, "seed": {"source": "configs/lora_sd15.yaml", "value": 42}, "train_batch_size": {"source": "configs/lora_sd15.yaml", "value": 1}, "train_text_encoder": {"source": "configs/lora_sd15.yaml", "value": false}, "trigger_word": {"source": "configs/lora_sd15.yaml", "value": "soybeanseed"}}
- Samples copied: 7
- Base vs. LoRA evidence: N/A
- Visual evaluation status: N/A

## System Demo

- Demo cases: 5
- Successful cases: 5
- Mean visual inference seconds: 8.850886
- Mean retrieval seconds: 0.809960
- Mean total seconds: 9.671234

## Missing Data

- Human visual evaluation of LoRA samples is not available.
- No base-vs-LoRA comparison evidence is available.
- No base-vs-LoRA comparison images were found.
- No external run manifest was found.
- Notebook outputs/logs are missing; hardware and training time are not verified.
- Parameter not verified: hardware
- Parameter not verified: training_time
- RAG human-review metrics are pending.

## Failures

- rag / RAG009: Expected document was not retrieved in top 5. (results/rag/evaluation/query_results.csv)
- rag / RAG012: Expected document was not retrieved in top 5. (results/rag/evaluation/query_results.csv)
- rag / RAG017: Expected document was not retrieved in top 5. (results/rag/evaluation/query_results.csv)
- rag / RAG020: Expected document was not retrieved in top 5. (results/rag/evaluation/query_results.csv)

## Generated Files

- `results/system/final_metrics.json`
- `results/system/final_metrics.csv`
- `results/system/demo_cases.csv`
- `results/system/latency_report.csv`
- `results/system/failure_cases.csv`
- `results/system/final_evaluation_report.md`
