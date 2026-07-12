# Alcance y arquitectura

## Entrada

Fotografía individual de una semilla de soja en condiciones razonables de iluminación y enfoque.

## Salidas

- Categoría visual estimada.
- Probabilidad o confianza.
- Evidencia documental recuperada.
- Posibles causas descritas como hipótesis documentadas.
- Medidas de prevención, almacenamiento o manejo con fuentes.
- Limitaciones y advertencia de revisión humana.

## Fuera de alcance

- Confirmar un patógeno específico sin análisis especializado.
- Analizar lotes completos en tiempo real.
- Sustituir decisiones de un profesional.

## Componentes

1. Ingesta y preprocesamiento de imagen.
2. Clasificador visual ajustado.
3. Constructor de consulta.
4. Recuperador RAG.
5. Generador de informe.
6. Módulo experimental SD 1.5 + LoRA.
7. Interfaz Streamlit.
