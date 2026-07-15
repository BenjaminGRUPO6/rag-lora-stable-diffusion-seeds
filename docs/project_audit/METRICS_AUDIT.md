# Auditoria de metricas

Auditoria: 2026-07-14T02:12:47-05:00

## Fuentes
|path|type|size_bytes|status|
|---|---|---|---|
|results/system/final_metrics.json|final_metrics.json|7949|readable|
|results/vision/resnet18_baseline/archive_before_reconciliation/classification_report.csv|classification_report.csv|622|readable|
|results/vision/resnet18_baseline/archive_before_reconciliation/metrics_test.json|metrics_test.json|957|readable|
|results/vision/resnet18_baseline/archive_before_reconciliation/metrics_validation.json|metrics_validation.json|930|readable|
|results/vision/resnet18_baseline/archive_before_reconciliation/run_summary.json|run_summary.json|714|readable|
|results/vision/resnet18_baseline/classification_report.csv|classification_report.csv|622|readable|
|results/vision/resnet18_baseline/metrics_test.json|metrics_test.json|957|readable|
|results/vision/resnet18_baseline/metrics_validation.json|metrics_validation.json|930|readable|
|results/vision/resnet18_baseline/run_summary.json|run_summary.json|786|readable|
|results/vision/resultados_1_baseline/archive_original/classification_report.csv|classification_report.csv|622|readable|
|results/vision/resultados_1_baseline/archive_original/metrics_test.json|metrics_test.json|957|readable|
|results/vision/resultados_1_baseline/archive_original/metrics_validation.json|metrics_validation.json|930|readable|
|results/vision/resultados_1_baseline/archive_original/r1_metricas.json|r1_metricas.json|957|readable|
|results/vision/resultados_1_baseline/archive_original/run_summary.json|run_summary.json|786|readable|
|results/vision/resultados_1_baseline/classification_report.csv|classification_report.csv|622|readable|
|results/vision/resultados_1_baseline/metrics_test.json|metrics_test.json|957|readable|
|results/vision/resultados_1_baseline/metrics_validation.json|metrics_validation.json|930|readable|
|results/vision/resultados_1_baseline/r1_metricas.json|r1_metricas.json|957|readable|
|results/vision/resultados_1_baseline/run_summary.json|run_summary.json|1608|readable|
|results/vision/resultados_2_mejoras/05_resnet18_v2/classification_report.csv|classification_report.csv|591|readable|
|results/vision/resultados_2_mejoras/05_resnet18_v2/metrics_test.json|metrics_test.json|926|readable|
|results/vision/resultados_2_mejoras/05_resnet18_v2/metrics_validation.json|metrics_validation.json|952|readable|
|results/vision/resultados_2_mejoras/05_resnet18_v2/run_summary.json|run_summary.json|1208|readable|
|results/vision/resultados_2_mejoras/07_tta/tta_test_results.json|tta_test_results.json|1461|readable|
|results/vision/resultados_2_mejoras/08_comparacion_modelos/run_summary.json|run_summary.json|3000|readable|
|results/vision/resultados_2_mejoras/08_comparacion_modelos/smoke_test/run_summary.json|run_summary.json|1209|readable|
|results/vision/resultados_2_mejoras/final/final_metrics.json|final_metrics.json|21612|readable|
|results/vision/smoke_test/classification_report.csv|classification_report.csv|297|readable|
|results/vision/smoke_test/metrics_test.json|metrics_test.json|707|readable|
|results/vision/smoke_test/metrics_validation.json|metrics_validation.json|739|readable|


R1 accuracy `0.6704980842911877`, macro-F1 `0.6259550750897566`. R2 final accuracy `0.9176245210727969`, macro-F1 `0.9168669642726247`. Soportes: R1 `522`, R2 `522`.

## Discrepancias
|file_a|value_a|file_b|value_b|possible_cause|status|
|---|---|---|---|---|---|
|results/vision/resultados_1_baseline/run_summary.json|validation reconciliada|archive_original metrics|valor obsoleto|R1 usa la validacion reconciliada del checkpoint porque los valores altos archivados fueron marcados como obsoletos.|resuelto|

