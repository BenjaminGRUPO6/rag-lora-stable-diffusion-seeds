# Guía de colaboración

## Ramas

- `main`: versión estable.
- `feature/dataset-*`: preparación de datos.
- `feature/vision-*`: modelo visual.
- `feature/lora-*`: Stable Diffusion 1.5 y LoRA.
- `feature/rag-*`: recuperación documental.
- `feature/app-*`: interfaz e integración.
- `docs/*`: documentación.

## Flujo

1. Actualice `main`.
2. Cree una rama para una sola tarea.
3. Realice commits pequeños y descriptivos.
4. Ejecute pruebas.
5. Abra un Pull Request.
6. Otro integrante revisa antes del merge.

## Convención de commits

- `feat:` nueva funcionalidad.
- `fix:` corrección.
- `docs:` documentación.
- `test:` pruebas.
- `refactor:` reorganización sin cambiar comportamiento.
- `chore:` configuración o mantenimiento.

Ejemplo: `feat: agrega auditoría de duplicados del dataset`.
