# Resultados 1 — Baseline ResNet18

Generated at UTC: 2026-07-13T18:51:38+00:00

## Evaluacion canonica

- Checkpoint: `models/vision/resnet18_baseline_best.pt`
- Checkpoint SHA-256: `38fd4c1f566acf22f538d87adaadb79d7c5b1a062defd86c64e577da4e538f23`
- Manifest: `data/metadata/dataset_split.csv`
- Test images: 522
- Synthetic test images: 0
- Accuracy: 0.670498084291
- Macro precision: 0.741193380401
- Macro recall: 0.650534190888
- Macro-F1: 0.625955075090

## F1 por clase

| class | support | f1 |
| --- | ---: | ---: |
| intact | 91 | 0.296296296296 |
| spotted | 106 | 0.724409448819 |
| immature | 112 | 0.688405797101 |
| broken | 100 | 0.641975308642 |
| skin_damaged | 113 | 0.778688524590 |

## Causa de la discrepancia

- Archived run_summary.json reported test_macro_f1=0.917840678752, while archived metrics_test.json reported macro_f1=0.625955075090.
- Archived classification_report.csv agrees with metrics_test.json at macro-F1=0.625955075090.
- The new checkpoint evaluation exactly reproduces metrics_test.json and classification_report.csv, so the higher run_summary value is stale.
- Checkpoint metadata also contradicts the archived summary: checkpoint epoch=1, checkpoint best_validation_macro_f1=0.6605130367624343, summary epochs_ran=11, summary best_validation_macro_f1=0.9231883922779236.
