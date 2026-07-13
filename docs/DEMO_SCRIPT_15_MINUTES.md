# Guion de demo de 15 minutos

## 0:00-2:00 Problema empresarial

Mensaje clave: la revision visual de semillas de soja puede ser lenta, variable y dificil de respaldar documentalmente.

Decir:

- "El objetivo no es diagnosticar enfermedades, sino clasificar defectos visibles."
- "El sistema trabaja con cinco categorias: `intact`, `spotted`, `immature`, `broken`, `skin_damaged`."
- "`spotted` no equivale a hongo; es una categoria visual."
- "La salida se acompana de evidencia documental y limitaciones."

## 2:00-5:00 Arquitectura y justificacion

Mostrar `docs/FINAL_ARCHITECTURE.md` o un diagrama simple.

Puntos:

- ResNet18 estima categoria y confianza.
- Una regla de incertidumbre marca predicciones con baja confianza o margen estrecho.
- RAG recupera documentos tecnicos con FAISS y embeddings `all-MiniLM-L6-v2`.
- El informe es preliminar y usa fragmentos recuperados.
- LoRA fue entrenado como evidencia generativa, pero el Experimento B fue aplazado.

Frase sugerida: "Separamos prediccion visual de evidencia documental para no convertir una etiqueta visual en diagnostico."

## 5:00-10:00 Demostracion

Preparacion antes de iniciar:

```powershell
python scripts/run_demo.py --serve
```

Pasos en la app:

1. Cargar una imagen de validacion o una imagen preparada.
2. Ejecutar analisis.
3. Mostrar clase estimada, confianza e incertidumbre.
4. Abrir pestana de evidencia RAG.
5. Mostrar fragmentos, titulos y paginas cuando esten disponibles.
6. Abrir informe preliminar y leer limitaciones.
7. Mostrar seccion LoRA como evidencia, no como mejora demostrada.

Ruta de contingencia CLI:

```powershell
python scripts/analyze_seed.py `
  --image data/processed/validation/immature/1.jpg `
  --output results/reports/demo_cli_report.json `
  --device cpu
```

Si la prediccion falla, explicar: "El caso queda registrado como error o prediccion incierta; por eso el sistema exige revision humana."

## 10:00-13:00 Metricas y limitaciones

Mostrar `docs/FINAL_RESULTS.md`.

Metricas principales:

- Dataset: 5513 imagenes auditadas, 5223 incluidas.
- ResNet18 test: accuracy 0.670498, macro-F1 0.625955.
- F1 bajo de `intact`: 0.296296.
- RAG: Hit@5 0.80, MRR 0.589167, cuatro fallos en top 5.
- Demo: 5 casos exitosos, tiempo total medio 9.671234 s.
- LoRA: adaptador entrenado, evidencia parcial.

Limitaciones que deben decirse:

- No diagnostica hongos ni enfermedades.
- Revision humana RAG pendiente.
- LoRA no tiene comparacion base vs. LoRA.
- Experimento B fue aplazado.

## 13:00-15:00 Etica y conclusiones

Mensaje final:

- "El valor del prototipo esta en combinar una clasificacion visual reproducible con evidencia documental trazable."
- "No reemplaza a un especialista."
- "La siguiente etapa responsable es mejorar la clase `intact`, revisar humanamente el RAG y evaluar sinteticos solo despues de aprobacion humana."

Cierre:

"El sistema demuestra integracion funcional de vision, RAG y evidencia LoRA, pero se presenta como prototipo academico con alcance limitado y no como herramienta diagnostica."
