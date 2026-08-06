# spikes/spike_a_enqueue_trial.py
"""Spike A: viabilidad de ``optuna.enqueue_trial`` para el arranque en caliente.

Valida por ejecución, y no por lectura del código, los supuestos sobre los que se
apoyan el pre-registro del protocolo experimental (v0.1) y el diseño del espacio de
búsqueda canónico:

1. Las configuraciones inyectadas se consumen como los primeros ensayos, en orden.
2. Los ensayos inyectados **cuentan** dentro del presupuesto de ensayos.
3. La secuencia es reproducible con la misma semilla.
4. Qué ocurre ante configuraciones inválidas (índice fuera de rango, clave
   inexistente, configuración parcial, entero de NumPy).
5. Los parámetros del estudio son **índices**, de modo que una misma configuración
   designa transformaciones distintas si cambia el orden de las opciones.
6. El ``MedianPruner`` no interviene cuando el objetivo no reporta valores
   intermedios.
7. El arranque en caliente alcanza el objetivo en menos ensayos que el frío sobre un
   problema controlado.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe spikes\\spike_a_enqueue_trial.py
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.transforms import GaussianBlur, MedianFilter

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

SEED = 20260806

# Espacio con la misma forma que el de ``OptunaSearchEngine``: un categórico por paso
# cuyos valores son índices, más un índice extra que representa "omitir el paso".
ESPACIO = {"paso_a": 3, "paso_b": 2, "paso_c": 4}


def _choices(n_opciones: int) -> list[int]:
    """Devuelve los índices válidos de un paso, incluido el de omisión.

    Args:
        n_opciones: Número de transformaciones candidatas del paso.

    Returns:
        Lista de índices ``0..n_opciones`` (el último representa omitir el paso).
    """
    return list(range(n_opciones + 1))


def _sugerir(trial: optuna.Trial) -> dict[str, int]:
    """Reproduce el muestreo de parámetros de ``OptunaSearchEngine``.

    Args:
        trial: Ensayo de Optuna en curso.

    Returns:
        Diccionario de nombre de paso a índice elegido.
    """
    return {clave: trial.suggest_categorical(clave, _choices(n)) for clave, n in ESPACIO.items()}


def _titulo(texto: str) -> None:
    """Imprime un encabezado de experimento."""
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def _veredicto(condicion: bool, texto: str) -> bool:
    """Imprime el veredicto de una verificación.

    Args:
        condicion: Resultado de la verificación.
        texto: Descripción de lo verificado.

    Returns:
        La misma condición, para poder acumular resultados.
    """
    print(f"  [{'OK ' if condicion else 'FALLA'}] {texto}")
    return condicion


# --------------------------------------------------------------------------- #
# 1 y 2. Orden de consumo y conteo dentro del presupuesto
# --------------------------------------------------------------------------- #
def experimento_orden_y_conteo() -> bool:
    """Verifica que lo inyectado se consuma primero, en orden, y cuente al presupuesto."""
    _titulo("1-2. Orden de consumo de los ensayos inyectados y conteo del presupuesto")

    inyectadas = [
        {"paso_a": 0, "paso_b": 1, "paso_c": 2},
        {"paso_a": 3, "paso_b": 0, "paso_c": 4},
        {"paso_a": 1, "paso_b": 2, "paso_c": 0},
        {"paso_a": 2, "paso_b": 1, "paso_c": 1},
        {"paso_a": 0, "paso_b": 0, "paso_c": 3},
    ]
    presupuesto = 10

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    for configuracion in inyectadas:
        study.enqueue_trial(configuracion)
    study.optimize(lambda trial: float(sum(_sugerir(trial).values())), n_trials=presupuesto)

    observadas = [t.params for t in study.trials[: len(inyectadas)]]
    print(f"  inyectadas: {inyectadas}")
    print(f"  primeras {len(inyectadas)} observadas: {observadas}")
    print(f"  ensayos totales: {len(study.trials)} (presupuesto solicitado: {presupuesto})")

    resultados = [
        _veredicto(observadas == inyectadas, "las inyectadas se consumen primero y en orden"),
        _veredicto(
            len(study.trials) == presupuesto,
            f"los inyectados cuentan al presupuesto: {presupuesto} ensayos en total, "
            f"no {presupuesto + len(inyectadas)}",
        ),
    ]
    return all(resultados)


# --------------------------------------------------------------------------- #
# 3. Reproducibilidad
# --------------------------------------------------------------------------- #
def _secuencia(con_inyeccion: bool) -> list[dict[str, int]]:
    """Ejecuta un estudio y devuelve la secuencia de parámetros de sus ensayos.

    Args:
        con_inyeccion: Si se inyecta una configuración inicial.

    Returns:
        Lista de diccionarios de parámetros, en orden de ensayo.
    """
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    if con_inyeccion:
        study.enqueue_trial({"paso_a": 1, "paso_b": 1, "paso_c": 1})
    study.optimize(lambda trial: float(sum(_sugerir(trial).values())), n_trials=12)
    return [t.params for t in study.trials]


def experimento_reproducibilidad() -> bool:
    """Verifica que la misma semilla produzca la misma secuencia de ensayos."""
    _titulo("3. Reproducibilidad con la misma semilla")

    primera, segunda = _secuencia(con_inyeccion=True), _secuencia(con_inyeccion=True)
    sin_inyeccion = _secuencia(con_inyeccion=False)

    resultados = [
        _veredicto(primera == segunda, "dos corridas con la misma semilla dan la misma secuencia"),
        _veredicto(
            primera != sin_inyeccion,
            "inyectar cambia la secuencia respecto de no inyectar (el efecto es real)",
        ),
    ]
    return all(resultados)


# --------------------------------------------------------------------------- #
# 4. Configuraciones inválidas
# --------------------------------------------------------------------------- #
def _intentar(descripcion: str, configuracion: dict) -> None:
    """Inyecta una configuración y reporta si el estudio la acepta o la rechaza.

    Args:
        descripcion: Texto que describe el caso probado.
        configuracion: Configuración a inyectar.
    """
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    with warnings.catch_warnings(record=True) as capturadas:
        warnings.simplefilter("always")
        try:
            study.enqueue_trial(configuracion)
            study.optimize(lambda trial: float(sum(_sugerir(trial).values())), n_trials=2)
            estado = study.trials[0].state.name
            observado = study.trials[0].params
            desenlace = f"aceptada, estado={estado}, params={observado}"
        except Exception as exc:  # noqa: BLE001 - el spike documenta cualquier fallo
            desenlace = f"EXCEPCION {type(exc).__name__}: {exc}"
    avisos = [str(w.message)[:90] for w in capturadas]
    print(f"  - {descripcion}\n      {desenlace}")
    if avisos:
        print(f"      avisos: {avisos}")


def experimento_configuraciones_invalidas() -> bool:
    """Documenta el comportamiento ante configuraciones que no encajan en el espacio."""
    _titulo("4. Configuraciones inválidas o parciales")

    _intentar("índice fuera de rango (paso_a=99, válidos 0..3)", {"paso_a": 99})
    _intentar("clave inexistente (paso_z)", {"paso_z": 1, "paso_a": 1})
    _intentar("configuración parcial (solo paso_a)", {"paso_a": 1})
    _intentar("entero de NumPy en vez de int", {"paso_a": np.int64(2)})
    print("\n  Nota: este experimento documenta comportamiento; no impone un veredicto.")
    return True


# --------------------------------------------------------------------------- #
# 5. Los parámetros son índices: ambigüedad entre espacios distintos
# --------------------------------------------------------------------------- #
def experimento_indices_ambiguos() -> bool:
    """Demuestra que una misma configuración designa transformaciones distintas.

    Es la justificación del espacio de búsqueda canónico y versionado: el índice no
    significa nada por sí mismo, depende del orden de la lista de opciones.
    """
    _titulo("5. Los parámetros son índices, no transformaciones")

    espacio_v1 = {"suavizado": [GaussianBlur(kernel_size=3), MedianFilter(kernel_size=3)]}
    espacio_v2 = {"suavizado": [MedianFilter(kernel_size=3), GaussianBlur(kernel_size=3)]}
    configuracion = {"suavizado": 0}

    pipeline_v1 = Pipeline([espacio_v1["suavizado"][configuracion["suavizado"]]])
    pipeline_v2 = Pipeline([espacio_v2["suavizado"][configuracion["suavizado"]]])

    print(f"  configuración inyectada: {configuracion}")
    print(f"  con el espacio v1 -> {pipeline_v1}")
    print(f"  con el espacio v2 -> {pipeline_v2}")

    return _veredicto(
        pipeline_v1.get_params() != pipeline_v2.get_params(),
        "la misma configuración produce pipelines distintos si cambia el orden de opciones",
    )


# --------------------------------------------------------------------------- #
# 6. El MedianPruner no interviene sin valores intermedios
# --------------------------------------------------------------------------- #
def experimento_pruner_inerte() -> bool:
    """Verifica que sin ``trial.report`` no haya ensayos podados."""
    _titulo("6. El MedianPruner es inerte sin valores intermedios")

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=SEED), pruner=MedianPruner()
    )
    study.optimize(lambda trial: float(sum(_sugerir(trial).values())), n_trials=20)

    estados = {t.state.name for t in study.trials}
    print(f"  estados observados: {estados}")
    return _veredicto("PRUNED" not in estados, "ningún ensayo fue podado")


# --------------------------------------------------------------------------- #
# 7. Frío contra caliente sobre un problema controlado
# --------------------------------------------------------------------------- #
OBJETIVO_OCULTO = {"paso_a": 2, "paso_b": 0, "paso_c": 3}


def _puntaje_picudo(parametros: dict[str, int]) -> float:
    """Objetivo con premio solo al acierto exacto (paisaje picudo).

    Cuenta coincidencias exactas con el óptimo oculto. Con el objetivo fijado en el
    95 % del mejor valor, **solo la combinación exacta** lo alcanza: no hay crédito
    parcial.

    Args:
        parametros: Configuración propuesta.

    Returns:
        Número de coincidencias, con un desempate determinista pequeño.
    """
    coincidencias = sum(parametros[clave] == valor for clave, valor in OBJETIVO_OCULTO.items())
    return coincidencias + sum(parametros.values()) * 1e-3


def _puntaje_suave(parametros: dict[str, int]) -> float:
    """Objetivo con crédito parcial por cercanía (paisaje suave).

    Se parece más a una métrica real como macro-F1: configuraciones cercanas al
    óptimo obtienen valores cercanos al mejor, de modo que varias combinaciones
    alcanzan el 95 % del máximo.

    Args:
        parametros: Configuración propuesta.

    Returns:
        Suma de las cercanías normalizadas a cada componente del óptimo.
    """
    total = 0.0
    for clave, objetivo in OBJETIVO_OCULTO.items():
        alcance = ESPACIO[clave]  # índices válidos: 0..alcance
        total += 1.0 - abs(parametros[clave] - objetivo) / alcance
    return total


def _ensayos_hasta_objetivo(valores: list[float], objetivo: float) -> int:
    """Número de ensayos hasta alcanzar el objetivo, con censura en presupuesto+1.

    Args:
        valores: Valores de la métrica por ensayo, en orden.
        objetivo: Valor a alcanzar.

    Returns:
        El índice (base 1) del primer ensayo que alcanza el objetivo, o
        ``len(valores) + 1`` si nunca se alcanza (caso censurado).
    """
    for posicion, valor in enumerate(valores, start=1):
        if valor >= objetivo:
            return posicion
    return len(valores) + 1


PRESUPUESTO = 30
SEMILLAS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)

# Vecinas plausibles: coinciden con el óptimo en 2 de 3 posiciones, ninguna en las 3.
VECINAS = [
    {"paso_a": 2, "paso_b": 0, "paso_c": 0},
    {"paso_a": 2, "paso_b": 1, "paso_c": 3},
    {"paso_a": 0, "paso_b": 0, "paso_c": 3},
    {"paso_a": 2, "paso_b": 2, "paso_c": 3},
    {"paso_a": 1, "paso_b": 0, "paso_c": 3},
]


def _correr(
    objetivo: Callable[[dict[str, int]], float],
    inyectadas: list[dict[str, int]],
    semilla: int,
    n_startup_trials: int = 10,
) -> list[float]:
    """Ejecuta un estudio y devuelve la serie de valores por ensayo.

    Args:
        objetivo: Función de puntaje sobre una configuración.
        inyectadas: Configuraciones a inyectar como ensayos iniciales.
        semilla: Semilla del muestreador.
        n_startup_trials: Ensayos aleatorios antes de que TPE empiece a modelar.

    Returns:
        Lista de valores del objetivo, en orden de ensayo.
    """
    sampler = TPESampler(seed=semilla, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for configuracion in inyectadas:
        study.enqueue_trial(configuracion)
    study.optimize(lambda trial: objetivo(_sugerir(trial)), n_trials=PRESUPUESTO)
    return [t.value for t in study.trials]


def _configuraciones_aleatorias(semilla: int) -> list[dict[str, int]]:
    """Genera cinco configuraciones al azar: control de recomendador aleatorio.

    Args:
        semilla: Semilla del generador.

    Returns:
        Cinco configuraciones válidas del espacio.
    """
    rng = np.random.default_rng(semilla)
    return [
        {clave: int(rng.integers(0, n + 1)) for clave, n in ESPACIO.items()} for _ in range(5)
    ]


def _comparar_brazos(
    objetivo: Callable[[dict[str, int]], float], etiqueta: str
) -> dict[str, float]:
    """Compara los brazos de arranque sobre un objetivo dado.

    Args:
        objetivo: Función de puntaje sobre una configuración.
        etiqueta: Nombre del paisaje, para el encabezado de la tabla.

    Returns:
        Mediana de ensayos hasta el objetivo por brazo.
    """
    con_optimo = [dict(OBJETIVO_OCULTO), *VECINAS[:4]]
    brazos: dict[str, list[int]] = {
        "frío": [],
        "caliente (vecinas)": [],
        "caliente (aleatorio)": [],
        "caliente (con óptimo)": [],
        "caliente (vecinas, startup=5)": [],
    }

    for semilla in SEMILLAS:
        frio = _correr(objetivo, [], semilla)
        meta = 0.95 * max(frio)
        aleatorias = _configuraciones_aleatorias(semilla)
        brazos["frío"].append(_ensayos_hasta_objetivo(frio, meta))
        brazos["caliente (vecinas)"].append(
            _ensayos_hasta_objetivo(_correr(objetivo, VECINAS, semilla), meta)
        )
        brazos["caliente (aleatorio)"].append(
            _ensayos_hasta_objetivo(_correr(objetivo, aleatorias, semilla), meta)
        )
        brazos["caliente (con óptimo)"].append(
            _ensayos_hasta_objetivo(_correr(objetivo, con_optimo, semilla), meta)
        )
        brazos["caliente (vecinas, startup=5)"].append(
            _ensayos_hasta_objetivo(_correr(objetivo, VECINAS, semilla, n_startup_trials=5), meta)
        )

    censura = PRESUPUESTO + 1
    print(f"\n  --- paisaje {etiqueta} " + "-" * (52 - len(etiqueta)))
    print(f"  {'brazo':<32}{'mediana':>9}{'censur.':>9}  ensayos")
    for nombre, valores in brazos.items():
        censurados = sum(valor == censura for valor in valores)
        print(f"  {nombre:<32}{float(np.median(valores)):>9.1f}{censurados:>9}  {valores}")

    return {nombre: float(np.median(valores)) for nombre, valores in brazos.items()}


def experimento_frio_vs_caliente() -> bool:
    """Compara formas de arranque sobre dos paisajes: picudo y suave.

    Los brazos son: frío; caliente con vecinas que no contienen el óptimo; caliente
    con un recomendador aleatorio (control); caliente con el óptimo incluido; y
    caliente con vecinas reduciendo ``n_startup_trials`` para que TPE empiece a
    modelar apenas terminan los ensayos inyectados.
    """
    _titulo("7. Frío contra caliente sobre un problema controlado")
    print(f"  {len(SEMILLAS)} semillas, presupuesto {PRESUPUESTO}, censura en {PRESUPUESTO + 1}")

    picudo = _comparar_brazos(_puntaje_picudo, "PICUDO (solo el acierto exacto alcanza la meta)")
    suave = _comparar_brazos(_puntaje_suave, "SUAVE (crédito parcial por cercanía)")

    print()
    mecanismo = _veredicto(
        picudo["caliente (con óptimo)"] < picudo["frío"]
        and suave["caliente (con óptimo)"] <= suave["frío"],
        "mecanismo: con recomendaciones buenas, el caliente reduce los ensayos",
    )

    print("\n  HALLAZGOS (no son fallos del mecanismo, son resultados):")
    for etiqueta, medianas in (("picudo", picudo), ("suave", suave)):
        peor = medianas["caliente (vecinas)"] > medianas["frío"]
        peor_que_azar = medianas["caliente (vecinas)"] > medianas["caliente (aleatorio)"]
        print(
            f"    - paisaje {etiqueta}: vecinas incorrectas "
            f"{'EMPEORAN' if peor else 'no empeoran'} respecto del frío; "
            f"{'peores' if peor_que_azar else 'no peores'} que un recomendador aleatorio."
        )
    return mecanismo


def main() -> None:
    """Ejecuta todos los experimentos del spike y resume los veredictos."""
    print("SPIKE A - optuna.enqueue_trial")
    print(f"optuna {optuna.__version__} | semilla base {SEED}")

    experimentos = {
        "1-2. orden y conteo": experimento_orden_y_conteo,
        "3. reproducibilidad": experimento_reproducibilidad,
        "4. configuraciones inválidas": experimento_configuraciones_invalidas,
        "5. índices ambiguos": experimento_indices_ambiguos,
        "6. pruner inerte": experimento_pruner_inerte,
        "7. frío vs caliente": experimento_frio_vs_caliente,
    }
    resultados = {nombre: funcion() for nombre, funcion in experimentos.items()}

    _titulo("RESUMEN")
    for nombre, resultado in resultados.items():
        print(f"  [{'OK ' if resultado else 'FALLA'}] {nombre}")


if __name__ == "__main__":
    main()
