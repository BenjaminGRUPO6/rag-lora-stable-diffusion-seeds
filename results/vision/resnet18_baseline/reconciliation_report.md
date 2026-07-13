# ResNet18 baseline reconciliation

Generated at UTC: 2026-07-13T06:42:31.668681+00:00

## Canonical evaluation

- Checkpoint: `models\vision\resnet18_baseline_best.pt`
- Manifest: `data\metadata\dataset_split.csv`
- Test images: 522
- Accuracy: 0.670498084291
- Macro-F1: 0.625955075090

## F1 by class

| class | support | f1 |
| --- | ---: | ---: |
| intact | 91 | 0.296296296296 |
| spotted | 106 | 0.724409448819 |
| immature | 112 | 0.688405797101 |
| broken | 100 | 0.641975308642 |
| skin_damaged | 113 | 0.778688524590 |

## Discrepancy explanation

- Archived run_summary.json reported test_macro_f1=0.917840678752, but archived metrics_test.json reported macro_f1=0.625955075090.
- Canonical evaluation matches the archived metrics_test.json and classification_report.csv, so the higher run_summary value was stale.
- Checkpoint metadata is incompatible with archived run_summary.json: checkpoint epoch=1, checkpoint best_validation_macro_f1=0.6605130367624343, summary epochs_ran=11, summary best_validation_macro_f1=0.9231883922779236.
