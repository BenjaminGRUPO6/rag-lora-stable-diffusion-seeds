# Configuración de GitHub desde cero

## Crear el repositorio

1. En GitHub seleccione **New repository**.
2. Nombre: `rag-lora-stable-diffusion-seeds`.
3. Visibilidad inicial: privada.
4. No inicialice con archivos si va a subir esta carpeta completa.
5. Invite a los integrantes desde **Settings > Collaborators**.

## Publicar esta estructura

```powershell
cd RUTA\rag-lora-stable-diffusion-seeds
git init
git branch -M main
git add .
git commit -m "chore: configura estructura final de SeedCare-RAG LoRA"
git remote add origin URL_DEL_REPOSITORIO
git push -u origin main
```

## Trabajo diario

```powershell
git checkout main
git pull origin main
git checkout -b feature/nombre-tarea
# editar y probar
git add ARCHIVOS_ESPECIFICOS
git commit -m "feat: descripcion concreta"
git push -u origin feature/nombre-tarea
```

Abra un Pull Request hacia `main` y solicite revisión de otro integrante.

## No versionar

- datos completos;
- credenciales;
- checkpoints;
- LoRA `.safetensors`;
- índices FAISS;
- salidas temporales.
