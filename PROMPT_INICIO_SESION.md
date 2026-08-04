# Prompt de inicio de sesión — AutoPipe (TFG)

> Copiar y pegar al comienzo de cada sesión nueva de Claude Code en este repositorio.

---

**Antes de responder cualquier solicitud de código, lee `CLAUDE.md` en la raíz del
repositorio.** Contiene la arquitectura, las interfaces públicas, los criterios de éxito
comprometidos y la lista de cosas que no debes hacer sin confirmarme. No propongas ni
escribas código sin haberlo leído en esta sesión.

## Contexto

Estoy desarrollando **AutoPipe**, mi Trabajo Final de Graduación de Ingeniería en
Computadores (TEC): generalizo la librería `AutoPipeAD` —que seleccionaba automáticamente
pipelines de preprocesamiento para detección de anomalías en imágenes— hacia un framework
agnóstico a la tarea con soporte de datos tabulares. Sobre esa base añado una capa de
meta-aprendizaje que, a partir de las meta-características de un dataset, recomienda
configuraciones iniciales de búsqueda (*warm-start*) para Optuna. La hipótesis a validar
es si ese arranque en caliente reduce el costo de la búsqueda frente al arranque en frío,
medido con validación *leave-one-dataset-out*.

## Estado del repositorio

Migré desde el proyecto base los 6 módulos de `autopipe/` (2 158 LOC), la suite de 134
pruebas (97 % de cobertura), 3 notebooks de ejemplo y la documentación de contexto
(anteproyecto del TFG y paper del proyecto base en `docs/`). El paquete fue renombrado de
`autopipead` a `autopipe`. Todo lo relacionado con meta-aprendizaje —extractor de
meta-características, base de meta-conocimiento, recomendador y el harness experimental—
está **por construir desde cero**.

## Cómo quiero trabajar

- **Tipo de solicitudes**: diseño e implementación de módulos nuevos, refactors del núcleo
  existente, pruebas, experimentos reproducibles y redacción/actualización de documentos
  LaTeX.
- **Antes de modificar una interfaz existente** (`BaseTransform`, la firma `run(...)` de
  los motores, las claves de `get_params()` o de las métricas): avísame primero, explica
  qué se rompe y espera mi confirmación. Los 134 tests migrados y los notebooks dependen
  de ellas.
- **Verifica, no asumas**: si una afirmación se puede comprobar ejecutando los tests, el
  linter o un script, ejecútalo y muéstrame el resultado real. Si algo falla, dímelo con
  la salida; no lo maquilles.
- **Reporte de cambios**: al terminar, dime qué archivos tocaste y qué cambió en cada uno,
  en una lista breve. Nada de resúmenes genéricos.
- **Alcance**: no amplíes el alcance definido en el anteproyecto ni cambies los criterios
  de éxito de la sección 4.3 por iniciativa propia. Si crees que algo debería cambiar,
  propónmelo con argumentos y espera respuesta.
- **Commits**: solo cuando yo lo pida. Mensajes convencionales en inglés, **sin líneas de
  coautoría, firmas automáticas ni referencias al asistente**. Nunca hagas `push` sin
  pedírmelo.
- **Trabajo por partes**: si una tarea tiene varias fases, confírmame el resultado de cada
  una antes de seguir con la siguiente.
