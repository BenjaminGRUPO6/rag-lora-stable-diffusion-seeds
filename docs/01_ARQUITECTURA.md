# Arquitectura

## Módulos

1. **Datos:** auditoría, etiquetas, split y aumento.
2. **Visión:** fine-tuning de un clasificador.
3. **Generación:** entrenamiento LoRA de SD 1.5 para clases minoritarias.
4. **RAG:** indexación y recuperación de fuentes técnicas.
5. **Informe:** salida estructurada con fuentes y advertencia.
6. **Aplicación:** interfaz Streamlit.

## Flujo de inferencia

```text
Imagen -> clasificador -> etiqueta/confianza -> consulta RAG -> fuentes -> informe
```

## Flujo de entrenamiento

```text
Dataset real verificado
  |-> clasificador visual -> checkpoint y métricas
  |-> subconjunto minoritario + captions -> SD 1.5 LoRA -> sintéticos revisados
                                                    |
                                                    v
                                     nuevo entrenamiento comparativo
```
