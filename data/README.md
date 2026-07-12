# Organización de datos

El dataset completo no se versiona en GitHub.

## Etiquetas principales

- `healthy`: semilla sin daño visible relevante.
- `physical_damage`: fractura, perforación, abrasión, aplastamiento u otra lesión mecánica.
- `biological_damage`: posible hongo, pudrición, manchas compatibles con agente biológico o daño por plaga.
- `morphological_damage`: deformación, forma o tamaño anormal, desarrollo incompleto.
- `unclassified`: muestra pendiente de revisión.

## Regla de separación

La división `train/validation/test` se realiza antes de crear copias aumentadas o sintéticas. Las imágenes sintéticas se incorporan únicamente a `train` después de revisión humana.

## Documentos para RAG

Cada documento debe registrar título, institución/autores, año, tipo, idioma, procedencia y temas cubiertos. No use documentos sin procedencia verificable.
