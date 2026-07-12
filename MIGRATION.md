# Migración desde la estructura anterior

La estructura anterior trataba a Stable Diffusion como salida principal. La versión actual corrige el objetivo:

- Núcleo: clasificación visual + RAG + informe con fuentes.
- Entrenamiento 1: fine-tuning del clasificador visual.
- Entrenamiento 2: Stable Diffusion 1.5 con LoRA.
- Uso de LoRA: experimento de ampliación sintética, no diagnóstico.

## Actualizar un repositorio existente

```powershell
git checkout main
git pull origin main
git checkout -b backup/estructura-anterior
git push -u origin backup/estructura-anterior

git checkout main
git checkout -b refactor/seedcare-soybean-final
```

Extrae el ZIP en una carpeta temporal, copia su contenido a la raíz del repositorio y reemplaza los archivos. Después:

```powershell
git status
git add -A
git commit -m "refactor: adapta el proyecto a SeedCare-RAG LoRA"
git push -u origin refactor/seedcare-soybean-final
```

Crea un Pull Request hacia `main`. Antes del commit verifica que no aparezcan `data/raw/`, `.env`, modelos, checkpoints ni índices vectoriales.
