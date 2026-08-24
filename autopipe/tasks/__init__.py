"""
tasks -- learning task definitions and metric registry.

Public API:
    Task                abstract description of a learning task
    ANOMALY_DETECTION   built-in binary anomaly detection task
    CLASSIFICATION      built-in binary/multiclass classification task
    TASKS               registry of built-in tasks, keyed by name
    get_task            resolve a task from its name or instance
"""

from __future__ import annotations

from autopipe.tasks.anomaly import ANOMALY_DETECTION
from autopipe.tasks.base import Task
from autopipe.tasks.classification import CLASSIFICATION

TASKS: dict[str, Task] = {
    task.name: task for task in (ANOMALY_DETECTION, CLASSIFICATION)
}

DEFAULT_TASK = ANOMALY_DETECTION
"""Tarea usada cuando no se especifica ninguna, por continuidad con la
librería base."""


def get_task(task: str | Task) -> Task:
    """Resuelve una tarea a partir de su nombre o de la propia instancia.

    Aceptar también instancias permite que una persona registre una tarea
    propia sin modificar la librería.

    Args:
        task: Nombre de una tarea del registro, o una instancia de :class:`Task`.

    Returns:
        La tarea correspondiente.

    Raises:
        ValueError: Si el nombre no está en el registro.
        TypeError: Si el argumento no es una cadena ni una :class:`Task`.
    """
    if isinstance(task, Task):
        return task
    if not isinstance(task, str):
        raise TypeError(
            f"La tarea debe ser un nombre o una instancia de Task, se recibió "
            f"{type(task).__name__}."
        )
    try:
        return TASKS[task]
    except KeyError as exc:
        raise ValueError(
            f"Tarea no soportada: {task!r}. Use una de {tuple(TASKS)}."
        ) from exc


__all__ = [
    "Task",
    "ANOMALY_DETECTION",
    "CLASSIFICATION",
    "TASKS",
    "DEFAULT_TASK",
    "get_task",
]
