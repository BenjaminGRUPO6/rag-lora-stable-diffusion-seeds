# Uso de Codex local

Codex debe iniciarse desde la raíz del repositorio para que lea `AGENTS.md`.

```powershell
codex
```

## Forma de trabajo

1. Asegurar que el trabajo actual esté en una rama.
2. Pedir primero inspección y plan.
3. Ejecutar una tarea delimitada.
4. Revisar `git diff`.
5. Ejecutar pruebas.
6. Corregir y documentar.
7. Hacer commit solo después de revisión humana.

Los archivos de `codex/tasks/` son encargos listos para copiar o adjuntar en la conversación de Codex. No deben ejecutarse todos a la vez.
