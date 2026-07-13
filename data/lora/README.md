# LoRA dataset

This directory stores the Stable Diffusion 1.5 LoRA preparation artifacts.

Only real images from `data/processed/train/` are eligible for LoRA training. Images from
`data/processed/validation/` and `data/processed/test/` must never be used for this dataset.

Generated structure:

```text
data/lora/train/
  images/
  metadata.jsonl
  selection_report.csv
```

`metadata.jsonl` contains one JSON object per selected image:

```json
{"file_name":"images/intact_00001.jpg","text":"photo of a soyseed soybean seed, intact condition, clean surface, centered inspection photography"}
```

The trigger word is `soyseed`.

Captions preserve the original visual class labels: `intact`, `spotted`, `immature`, `broken`,
and `skin_damaged`. Captions must describe visible appearance only. They must not use diagnostic
terms such as `fungus`, `disease`, or `infection`; `spotted` is a visual category, not a fungal
diagnosis.

The prepared image dataset and LoRA training outputs are not uploaded to GitHub. Keep generated
images, `metadata.jsonl`, model weights, and training logs ignored.
