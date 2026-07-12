# Organización de datos

El dataset completo no se versiona en GitHub.

## Etiquetas principales

- `intact`: semilla de soja sin defecto visible relevante.
- `spotted`: semilla de soja con manchas visibles; no confirma hongo ni enfermedad.
- `immature`: semilla de soja inmadura o con desarrollo incompleto.
- `broken`: semilla de soja rota o fracturada.
- `skin_damaged`: semilla de soja con daño visible en la cubierta.

## Regla de separación

La división `train/validation/test` se realiza antes de crear copias aumentadas o sintéticas. Las imágenes sintéticas se incorporan únicamente a `train` después de revisión humana.

## Documentos para RAG

Cada documento debe registrar título, institución/autores, año, tipo, idioma, procedencia y temas cubiertos. No use documentos sin procedencia verificable.
