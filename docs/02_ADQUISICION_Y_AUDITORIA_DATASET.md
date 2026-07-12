# Adquisición y auditoría del dataset

## Fuente principal prevista

Soybean Seeds, versión 6, Mendeley Data, DOI `10.17632/v6vzvfszj6.6`.

## Orden obligatorio

1. Descargar y conservar el archivo original como respaldo.
2. Registrar fecha y procedencia.
3. Colocar las carpetas dentro de `data/raw/soybean_seeds/`.
4. Verificar cinco clases esperadas.
5. Auditar archivos corruptos, dimensiones y duplicados.
6. Revisar manualmente resultados.
7. Crear `train`, `validation` y `test` con semilla fija.
8. Aplicar aumento solo a `train`.

## Comandos

```powershell
python scripts/verify_dataset_structure.py --dataset data/raw/soybean_seeds
python scripts/audit_dataset.py --dataset data/raw/soybean_seeds --output results/dataset_audit
```
