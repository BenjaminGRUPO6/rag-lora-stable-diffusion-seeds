# Etica y limitaciones

## Declaracion de alcance

SeedCare-RAG LoRA es una herramienta de apoyo visual y documental. No diagnostica enfermedades, no identifica patogenos, no reemplaza laboratorio y no debe usarse como unica base para decisiones agronomicas, comerciales o fitosanitarias.

## Riesgos principales

- Confundir una categoria visual con una causa biologica.
- Tomar `spotted` como diagnostico de hongo.
- Sobreinterpretar la confianza del clasificador.
- Presentar fragmentos recuperados como recomendacion concluyente.
- Usar imagenes sinteticas sin revision humana.
- Ocultar que el corpus documental es limitado y tiene metadatos incompletos.

## Medidas aplicadas

- La app muestra una advertencia de no diagnostico.
- El informe incluye limitaciones por defecto.
- `spotted` se declara como categoria visual.
- Las fuentes recuperadas se muestran con titulo, pagina y fragmento cuando estan disponibles.
- El RAG abstiene o limita recomendaciones cuando no hay evidencia suficiente.
- Los datos sinteticos solo pueden entrar a `train` despues de revision humana.
- `validation` y `test` se mantienen con imagenes reales.
- Pesos, datasets, indices y secretos no se versionan.

## Limitaciones empiricas

- ResNet18 alcanza `macro-F1=0.625955` en test, con F1 bajo para `intact` (`0.296296`).
- La evaluacion RAG tiene Hit@5 de `0.80`, pero cuatro consultas no recuperan el documento esperado en top 5.
- La revision humana del RAG esta pendiente.
- LoRA tiene evidencia local parcial: no hay logs, hardware, tiempo de entrenamiento ni comparacion base vs. LoRA.
- No hay metricas de aceptacion humana de imagenes sinteticas.
- El Experimento B fue aplazado y no existe comparacion del clasificador con datos sinteticos.

## Uso responsable en sustentacion

Presentar el sistema como prototipo academico reproducible. Evitar frases como "detecta hongos", "diagnostica enfermedad", "recomienda tratamiento" o "mejora comprobada por sinteticos". Usar formulaciones como "clasifica defectos visibles", "recupera evidencia documental" y "genera un informe preliminar con limitaciones".

## Trabajo futuro etico

- Revisar humanamente consultas RAG y salidas del informe.
- Completar metadatos documentales solo desde fuentes verificadas.
- Validar con expertos antes de uso operativo.
- Medir sesgos por iluminacion, fondo, camara y procedencia del dataset.
- Documentar aceptacion/rechazo de sinteticos con criterios explicitos.
