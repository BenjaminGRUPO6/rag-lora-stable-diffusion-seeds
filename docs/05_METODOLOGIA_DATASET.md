# Metodología del dataset

## Principios

- La calidad de etiqueta es más importante que multiplicar imágenes.
- Se conserva `original_id` para relacionar originales, aumentos y sintéticos.
- Validación y test contienen solo imágenes reales no vistas.
- Las imágenes sintéticas nunca se usan para evaluar.
- Las etiquetas de enfermedad se expresan como señales o posibles daños, no como diagnóstico confirmado.

## Aumento clásico

Rotación suave, volteo cuando la orientación no sea relevante, variación ligera de brillo/contraste y recorte controlado. Evitar cambios fuertes de color, deformaciones intensas y recortes que eliminen el daño visible.

## Datos sintéticos

- Generar solo para clases minoritarias.
- Revisar calidad y fidelidad de etiqueta.
- Registrar prompt, seed, LoRA y revisor.
- Rechazar imágenes ambiguas o irreales.
