# Preguntas probables del panel

## 1. El sistema diagnostica enfermedades?

No. Clasifica defectos visibles y recupera evidencia documental. No identifica patogenos ni reemplaza laboratorio.

## 2. `spotted` significa hongo?

No. `spotted` es una categoria visual del dataset. Puede estar relacionada con alteraciones visibles descritas por documentos, pero no confirma una causa.

## 3. Que modelos fueron entrenados?

Se entreno un baseline ResNet18 para clasificacion visual y se entreno un adaptador Stable Diffusion 1.5 + LoRA. El segundo ResNet18 con sinteticos, Experimento B, fue aplazado.

## 4. Cual es la metrica final del clasificador?

En test: accuracy 0.670498 y macro-F1 0.625955 sobre 522 imagenes. La clase mas debil fue `intact`, con F1 0.296296.

## 5. Por que no usan el macro-F1 alto que aparece en un resumen anterior?

Porque `results/vision/resnet18_baseline/reconciliation_report.md` lo marca como obsoleto. La metrica canonica es la de `metrics_test.json`, que coincide con la reconciliacion.

## 6. Como evaluaron el RAG?

Con 20 consultas registradas en `data/metadata/rag_evaluation_queries.csv`. Se midio recuperacion: Hit@1 0.45, Hit@3 0.70, Hit@5 0.80 y MRR 0.589167. No se evaluo generacion por LLM.

## 7. El RAG usa fuentes reales?

Si. Usa documentos aceptados registrados en `data/metadata/document_sources.csv` e indexados en `vector_db/`. Algunos metadatos bibliograficos estan incompletos y no se inventaron.

## 8. Que limitaciones tiene LoRA?

Existe adaptador local de 6.12 MB y metadata de 1000 imagenes, pero faltan logs, hardware, tiempo de entrenamiento, comparacion base vs. LoRA y evaluacion visual humana.

## 9. Por que aplazaron el Experimento B?

Porque incorporar sinteticos al entrenamiento requiere revision humana previa y una comparacion controlada. Para la entrega final se priorizo consolidar ResNet18 real, RAG, integracion y evidencia LoRA sin introducir datos no revisados.

## 10. Como evitan fuga de datos?

`validation` y `test` se mantienen con imagenes reales. Los duplicados exactos se excluyeron antes del split y los sinteticos no entran a evaluacion.

## 11. Que pasa si el sistema se equivoca?

La salida muestra confianza e incertidumbre. El informe declara limitaciones y debe ser revisado por una persona. El prototipo no automatiza decisiones finales.

## 12. Cual es el aporte principal?

La integracion reproducible de clasificacion visual, recuperacion documental y reporte preliminar con fuentes, mas evidencia de entrenamiento LoRA sin afirmar beneficios aun no medidos.

## 13. Que harian con mas tiempo?

Mejorar la clase `intact`, revisar humanamente el RAG, completar metadatos documentales desde fuentes verificadas, evaluar visualmente LoRA y ejecutar Experimento B con sinteticos aceptados.

## 14. Que referencias usaron?

Solo referencias registradas en `docs/11_REFERENCIAS_BASE.md` y fuentes documentales en `data/metadata/document_sources.csv`.

## 15. La demo puede correr sin GPU?

Si, con CPU, aunque puede ser mas lenta. El comando CLI acepta `--device cpu`.
