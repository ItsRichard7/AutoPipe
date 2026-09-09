# scripts/build_corpus_e2.py
"""Compone el corpus del experimento 2: 42 conjuntos de datos (v2.0 §7.1).

El experimento 1 usó los 25 que el cribado seleccionó entre 52 candidatos. El
experimento 2 amplía a 42, y el número no es 52 por una razón que el pre-registro
subraya: de los 27 que quedaron fuera, **diez lo hicieron por la regla de familia**
y siguen fuera. Entran los diecisiete que solo quedaron por debajo del corte en el
orden por sensibilidad.

**Por qué la regla de familia importa más aquí que al elegir el corpus original.**
En una validación por exclusión de conjunto, admitir dos conjuntos de la misma
familia significa que, en el pliegue que excluye a uno, su casi gemelo permanece
entre los datos con los que se ajusta el recomendador. El vecino más cercano de la
consulta sería un hermano suyo, y el arranque en caliente parecería acertar por una
similitud que no es la que el método pretende explotar. El sesgo iría **a favor** de
la hipótesis.

Este guion no descarga ni evalúa nada: solo compone la lista, leyendo el documento
que el cribado ya produjo.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\build_corpus_e2.py
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SELECCIONADO = Path("docs/experimentos/CORPUS-seleccionado.md")
SALIDA = Path("docs/experimentos/CORPUS-E2.md")

TOTAL_ESPERADO = 42
"""Tamaño que el pre-registro v2.0 fija. Se comprueba, no se supone."""

# Fila de la tabla del corpus: | # | ID | `nombre` | IQR | Instancias |
FILA_CORPUS = re.compile(
    r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|"
)
# Fila de los no seleccionados: | ID | `nombre` | IQR | motivo |
FILA_DESCARTE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*(familia|orden)\s*\|"
)


def leer(ruta: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Lee del documento del cribado sus tres grupos de conjuntos.

    Args:
        ruta: Documento ``CORPUS-seleccionado.md``.

    Returns:
        Tupla ``(seleccionados, por_orden, por_familia)``.

    Raises:
        SystemExit: Si el archivo no existe o no tiene filas legibles.
    """
    if not ruta.is_file():
        raise SystemExit(f"No existe {ruta}. Ejecute antes scripts/screen_corpus.py.")

    seleccionados: list[dict] = []
    por_orden: list[dict] = []
    por_familia: list[dict] = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        corpus = FILA_CORPUS.match(linea)
        if corpus:
            seleccionados.append(
                {
                    "did": int(corpus.group(1)),
                    "nombre": corpus.group(2),
                    "iqr": float(corpus.group(3)),
                    "origen": "corpus del experimento 1",
                }
            )
            continue
        descarte = FILA_DESCARTE.match(linea)
        if descarte:
            registro = {
                "did": int(descarte.group(1)),
                "nombre": descarte.group(2),
                "iqr": float(descarte.group(3)),
                "origen": "incorporado en el experimento 2",
            }
            destino = por_familia if descarte.group(4) == "familia" else por_orden
            destino.append(registro)

    if not seleccionados:
        raise SystemExit(f"No se pudo leer ningún conjunto seleccionado de {ruta}.")
    return seleccionados, por_orden, por_familia


def componer(ruta: Path = SELECCIONADO) -> tuple[list[dict], list[dict]]:
    """Compone el corpus del experimento 2 y comprueba su tamaño.

    Args:
        ruta: Documento del cribado.

    Returns:
        Tupla ``(corpus, excluidos_por_familia)``, el corpus ordenado por
        sensibilidad descendente.

    Raises:
        SystemExit: Si el corpus resultante no tiene el tamaño pre-registrado.
    """
    seleccionados, por_orden, por_familia = leer(ruta)
    corpus = sorted(seleccionados + por_orden, key=lambda d: (-d["iqr"], d["did"]))

    if len(corpus) != TOTAL_ESPERADO:
        raise SystemExit(
            f"El corpus del experimento 2 debería tener {TOTAL_ESPERADO} conjuntos y "
            f"salieron {len(corpus)} ({len(seleccionados)} del experimento 1 más "
            f"{len(por_orden)} incorporados). El pre-registro v2.0 fija ese número, "
            f"así que la discrepancia se rechaza en lugar de continuar."
        )
    if any(d in corpus for d in por_familia):
        raise SystemExit(
            "Un conjunto excluido por la regla de familia entró en el corpus. En un "
            "leave-one-dataset-out eso deja un casi gemelo entrenando al recomendador "
            "en el pliegue que excluye a su pareja, y el sesgo iría a favor de la "
            "hipótesis."
        )
    return corpus, por_familia


def redactar(corpus: list[dict], por_familia: list[dict]) -> str:
    """Redacta el documento del corpus del experimento 2.

    El formato de la tabla es el mismo que lee ``build_meta_knowledge.py``, de modo
    que la base se construye sobre esta lista sin ninguna conversión intermedia.

    Args:
        corpus: Conjuntos del experimento 2.
        por_familia: Los que siguen excluidos por la regla de familia.

    Returns:
        El documento markdown.
    """
    filas = "\n".join(
        f"| {i} | {d['did']} | `{d['nombre']}` | {d['iqr']:.4f} | {d['origen']} |"
        for i, d in enumerate(corpus, 1)
    )
    excluidos = "\n".join(
        f"| {d['did']} | `{d['nombre']}` | {d['iqr']:.4f} |" for d in por_familia
    )
    incorporados = sum(1 for d in corpus if d["origen"].startswith("incorporado"))
    return f"""# Corpus del experimento 2 (42 conjuntos)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\build_corpus_e2.py`.

- **Generado:** {date.today().isoformat()}
- **Entrada:** `{SELECCIONADO.as_posix()}`
- **Gobierna:** pre-registro del experimento 2, v2.0, etiqueta `prereg-v2.0`
- **Composición:** {len(corpus) - incorporados} del experimento 1 + {incorporados}
  incorporados = **{len(corpus)}**

## Por qué 42 y no 52

El cribado descargó y verificó 52 candidatos y seleccionó 25. De los 27 restantes,
**{len(por_familia)} quedaron fuera por la regla de familia y siguen fuera**; entran
los {incorporados} que solo quedaron por debajo del corte en el orden por
sensibilidad.

En una validación por exclusión de conjunto, admitir dos conjuntos de la misma
familia deja al casi gemelo entrenando al recomendador justo en el pliegue que
excluye a su pareja. El vecino más cercano sería un hermano, y el arranque en
caliente parecería acertar por una similitud que no es la que el método pretende
explotar. **El sesgo iría a favor de la hipótesis**, que es la única dirección en la
que no se puede admitir ninguno.

Los criterios de inclusión del anteproyecto se mantienen sin alteración: los 42 ya
los cumplen. Lo único que se retira es el corte por sensibilidad, que existía para
quedarse con 25.

## Corpus

Ordenado por sensibilidad al preprocesamiento descendente, que es el orden que
produjo el cribado.

| # | OpenML ID | Nombre | IQR | Procedencia |
|---:|---:|---|---:|---|
{filas}

## Siguen excluidos por la regla de familia

| OpenML ID | Nombre | IQR |
|---:|---|---:|
{excluidos}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el corpus se compuso.
    """
    corpus, por_familia = componer()
    SALIDA.write_text(redactar(corpus, por_familia), encoding="utf-8")
    incorporados = sum(1 for d in corpus if d["origen"].startswith("incorporado"))
    logger.info(
        "Corpus del experimento 2: %d conjuntos (%d del experimento 1 + %d "
        "incorporados). Siguen excluidos %d por regla de familia. Escrito en %s",
        len(corpus),
        len(corpus) - incorporados,
        incorporados,
        len(por_familia),
        SALIDA,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
