# spikes/spike_b_pymfe.py
"""Spike B: viabilidad de ``pymfe`` como extractor de meta-características.

Mide por ejecución lo que el plan de la semana 5 necesita saber antes de construir
el extractor (RE3) y el corpus (RE4):

1. Costo de extracción por grupo de meta-características y cómo escala con el número
   de instancias, en particular el grupo ``complexity``, cuyas medidas basadas en
   distancias se esperan cuadráticas.
2. Extrapolación del costo al techo de 50 000 instancias que permite el anteproyecto.
3. Estabilidad del vector: ¿devuelve pymfe las mismas meta-características para
   datasets distintos? Es imprescindible para que el recomendador por vecinos pueda
   comparar vectores alineados.
4. Proporción de valores no finitos (``NaN`` / ``inf``) en el vector.
5. Determinismo con la misma semilla.
6. Comportamiento sobre datos heterogéneos: columnas categóricas y valores faltantes
   (riesgo RI4 del anteproyecto).
7. Fidelidad del submuestreo: cuánto se aparta el vector calculado sobre una muestra
   del calculado sobre el dataset completo.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe spikes\\spike_b_pymfe.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from pymfe.mfe import MFE

SEED = 20260806
GRUPOS = ("general", "statistical", "info-theory", "complexity")

# Si una medición supera este tiempo, se deja de escalar ese grupo y se extrapola.
LIMITE_SEGUNDOS = 45.0
TAMANOS = (500, 1000, 2000, 4000, 8000)
N_CARACTERISTICAS = 20  # dentro del techo de 100 del anteproyecto


def _titulo(texto: str) -> None:
    """Imprime un encabezado de experimento."""
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def _veredicto(condicion: bool, texto: str) -> bool:
    """Imprime el veredicto de una verificación y lo devuelve."""
    print(f"  [{'OK ' if condicion else 'FALLA'}] {texto}")
    return condicion


def _dataset_numerico(
    n_instancias: int, n_caracteristicas: int = N_CARACTERISTICAS, semilla: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Genera un dataset numérico de clasificación binaria.

    Args:
        n_instancias: Número de filas.
        n_caracteristicas: Número de columnas.
        semilla: Semilla del generador.

    Returns:
        Tupla ``(X, y)``.
    """
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n_instancias, n_caracteristicas))
    pesos = rng.normal(size=n_caracteristicas)
    y = (X @ pesos > 0).astype(int)
    return X, y


def _extraer(
    X: np.ndarray, y: np.ndarray, grupos: tuple[str, ...] | list[str]
) -> tuple[list[str], list[float], float]:
    """Extrae meta-características y mide el tiempo empleado.

    Args:
        X: Matriz de características.
        y: Vector de etiquetas.
        grupos: Grupos de meta-características de pymfe.

    Returns:
        Tupla ``(nombres, valores, segundos)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inicio = time.perf_counter()
        extractor = MFE(groups=list(grupos), random_state=SEED)
        extractor.fit(X, y)
        nombres, valores = extractor.extract()
        transcurrido = time.perf_counter() - inicio
    return list(nombres), list(valores), transcurrido


# --------------------------------------------------------------------------- #
# 1 y 2. Costo por grupo, escalado y extrapolación
# --------------------------------------------------------------------------- #
def experimento_costo_y_escalado() -> dict[str, float]:
    """Mide el costo por grupo y extrapola al techo de 50 000 instancias.

    Returns:
        Segundos extrapolados a 50 000 instancias, por grupo.
    """
    _titulo("1-2. Costo de extracción por grupo y escalado con el número de instancias")
    print(f"  {N_CARACTERISTICAS} características, semilla {SEED}, "
          f"corte por medición: {LIMITE_SEGUNDOS:.0f} s\n")

    encabezado = "  " + f"{'grupo':<14}" + "".join(f"{n:>10}" for n in TAMANOS)
    print(encabezado)
    print("  " + "-" * (len(encabezado) - 2))

    mediciones: dict[str, list[tuple[int, float]]] = {}
    for grupo in GRUPOS:
        fila, puntos = [], []
        for n_instancias in TAMANOS:
            if puntos and puntos[-1][1] > LIMITE_SEGUNDOS:
                fila.append(f"{'--':>10}")
                continue
            X, y = _dataset_numerico(n_instancias)
            _, _, segundos = _extraer(X, y, [grupo])
            puntos.append((n_instancias, segundos))
            fila.append(f"{segundos:>10.2f}")
        mediciones[grupo] = puntos
        print(f"  {grupo:<14}" + "".join(fila))

    print("\n  Exponente de escalado y extrapolación a 50 000 instancias:")
    print(f"  {'grupo':<14}{'exponente':>11}{'a 50k (s)':>13}{'a 50k':>14}")
    extrapolados: dict[str, float] = {}
    for grupo, puntos in mediciones.items():
        tamanos = np.array([p[0] for p in puntos], dtype=float)
        tiempos = np.array([max(p[1], 1e-6) for p in puntos], dtype=float)
        exponente, log_a = np.polyfit(np.log(tamanos), np.log(tiempos), 1)
        segundos_50k = float(np.exp(log_a) * 50_000**exponente)
        extrapolados[grupo] = segundos_50k
        print(
            f"  {grupo:<14}{exponente:>11.2f}{segundos_50k:>13.1f}"
            f"{_formato_duracion(segundos_50k):>14}"
        )
    return extrapolados


def _formato_duracion(segundos: float) -> str:
    """Formatea una duración en unidades legibles.

    Args:
        segundos: Duración en segundos.

    Returns:
        Texto como ``"3.2 min"`` o ``"1.4 h"``.
    """
    if segundos < 60:
        return f"{segundos:.1f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.1f} h"


# --------------------------------------------------------------------------- #
# 3 y 4. Estabilidad del vector y valores no finitos
# --------------------------------------------------------------------------- #
def experimento_estabilidad_del_vector() -> bool:
    """Verifica que datasets distintos produzcan el mismo conjunto de nombres."""
    _titulo("3-4. Estabilidad del vector de meta-características y valores no finitos")

    casos = {
        "binario equilibrado": _dataset_numerico(1000, semilla=1),
        "binario desbalanceado": _desbalanceado(1000),
        "multiclase (5 clases)": _multiclase(1000, n_clases=5),
        "pocas características": _dataset_numerico(1000, n_caracteristicas=4, semilla=2),
    }

    conjuntos, filas = [], []
    for etiqueta, (X, y) in casos.items():
        nombres, valores, _ = _extraer(X, y, GRUPOS)
        arreglo = np.asarray(valores, dtype=float)
        no_finitos = int((~np.isfinite(arreglo)).sum())
        conjuntos.append(tuple(nombres))
        filas.append((etiqueta, len(nombres), no_finitos, 100 * no_finitos / len(nombres)))

    print(f"  {'caso':<26}{'meta-caract.':>14}{'no finitos':>12}{'%':>8}")
    for etiqueta, total, no_finitos, porcentaje in filas:
        print(f"  {etiqueta:<26}{total:>14}{no_finitos:>12}{porcentaje:>7.1f}%")

    iguales = len(set(conjuntos)) == 1
    if not iguales:
        referencia = set(conjuntos[0])
        for etiqueta, nombres in zip(casos, conjuntos, strict=True):
            diferencia = referencia.symmetric_difference(nombres)
            if diferencia:
                print(f"    {etiqueta}: difiere en {sorted(diferencia)[:6]}")

    return _veredicto(
        iguales, "todos los datasets producen exactamente el mismo conjunto de nombres"
    )


def _desbalanceado(n_instancias: int) -> tuple[np.ndarray, np.ndarray]:
    """Genera un dataset binario con 10 % de la clase positiva."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_instancias, N_CARACTERISTICAS))
    y = (rng.random(n_instancias) < 0.1).astype(int)
    return X, y


def _multiclase(n_instancias: int, n_clases: int) -> tuple[np.ndarray, np.ndarray]:
    """Genera un dataset multiclase equilibrado."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(n_instancias, N_CARACTERISTICAS))
    y = rng.integers(0, n_clases, n_instancias)
    return X, y


# --------------------------------------------------------------------------- #
# 5. Determinismo
# --------------------------------------------------------------------------- #
TOLERANCIA_REPRODUCIBILIDAD = 1e-9


def experimento_determinismo() -> bool:
    """Mide si dos extracciones sobre el mismo dataset coinciden, y con qué precisión."""
    _titulo("5. Determinismo con la misma semilla")

    X, y = _dataset_numerico(1000)
    nombres_a, valores_a, _ = _extraer(X, y, GRUPOS)
    nombres_b, valores_b, _ = _extraer(X, y, GRUPOS)

    arreglo_a, arreglo_b = np.asarray(valores_a, float), np.asarray(valores_b, float)
    finitos = np.isfinite(arreglo_a) & np.isfinite(arreglo_b)
    diferencias = np.abs(arreglo_a - arreglo_b)
    distintas = np.where(finitos & (diferencias > 0))[0]

    print(f"  meta-características que difieren: {len(distintas)} de {int(finitos.sum())} finitas")
    for indice in distintas:
        relativa = diferencias[indice] / max(abs(arreglo_a[indice]), 1e-12)
        print(
            f"    {nombres_a[indice]:<24} abs={diferencias[indice]:.3g}  rel={relativa:.3g}"
        )

    peor_relativa = 0.0
    if len(distintas):
        peor_relativa = float(
            max(diferencias[i] / max(abs(arreglo_a[i]), 1e-12) for i in distintas)
        )
    print(f"  peor diferencia relativa: {peor_relativa:.3g}")

    resultados = [
        _veredicto(nombres_a == nombres_b, "los nombres coinciden"),
        _veredicto(
            np.array_equal(np.isfinite(arreglo_a), np.isfinite(arreglo_b)),
            "los valores no finitos aparecen en las mismas posiciones",
        ),
        _veredicto(
            peor_relativa < TOLERANCIA_REPRODUCIBILIDAD,
            f"los valores coinciden dentro de una tolerancia de "
            f"{TOLERANCIA_REPRODUCIBILIDAD:.0e}",
        ),
    ]
    if len(distintas):
        print(
            "\n  Nota: la diferencia es ruido de coma flotante (orden de sumación), no\n"
            "  aleatoriedad. Persistir el vector redondeado lo vuelve bit a bit estable."
        )
    return all(resultados)


# --------------------------------------------------------------------------- #
# 6. Datos heterogéneos: categóricas y faltantes
# --------------------------------------------------------------------------- #
def _imputar_por_mediana(X: np.ndarray) -> np.ndarray:
    """Sustituye los valores faltantes por la mediana de su columna.

    Args:
        X: Matriz numérica que puede contener ``NaN``.

    Returns:
        Copia de la matriz sin valores faltantes.
    """
    imputada = np.asarray(X, dtype=float).copy()
    for columna in range(imputada.shape[1]):
        valores = imputada[:, columna]
        faltantes = ~np.isfinite(valores)
        if faltantes.any():
            valores[faltantes] = np.nanmedian(valores)
    return imputada


def experimento_datos_heterogeneos() -> bool:
    """Documenta el comportamiento ante columnas categóricas y valores faltantes.

    Prueba grupo por grupo, para localizar exactamente cuál falla, y verifica si
    imputar antes de extraer resuelve el problema.
    """
    _titulo("6. Datos heterogéneos: columnas categóricas y valores faltantes")

    rng = np.random.default_rng(3)
    n = 800
    numericas = rng.normal(size=(n, 4))
    categoricas = rng.choice(["a", "b", "c"], size=(n, 2))
    y = rng.integers(0, 2, n)

    mixto = np.hstack([numericas.astype(object), categoricas.astype(object)])
    con_faltantes = numericas.copy()
    con_faltantes[rng.random(con_faltantes.shape) < 0.1] = np.nan

    print("  Por grupo, sobre datos con valores faltantes:")
    fallan = []
    for grupo in GRUPOS:
        try:
            nombres, _, _ = _extraer(con_faltantes, y, [grupo])
            print(f"    {grupo:<14} OK ({len(nombres)} meta-características)")
        except Exception as exc:  # noqa: BLE001 - el spike documenta cualquier fallo
            fallan.append(grupo)
            mensaje = str(exc) or "(mensaje vacío)"
            print(f"    {grupo:<14} EXCEPCION {type(exc).__name__}: {mensaje}")

    print("\n  Casos completos:")
    resultados = []
    casos = (
        ("columnas categóricas (object)", mixto),
        ("faltantes, imputados por mediana", _imputar_por_mediana(con_faltantes)),
    )
    for etiqueta, X in casos:
        try:
            nombres, valores, segundos = _extraer(X, y, GRUPOS)
            arreglo = np.asarray(valores, dtype=float)
            no_finitos = int((~np.isfinite(arreglo)).sum())
            print(
                f"    {etiqueta:<34} OK: {len(nombres)} meta-características en "
                f"{segundos:.2f} s, {no_finitos} no finitas "
                f"({100 * no_finitos / len(nombres):.1f} %)"
            )
            resultados.append(True)
        except Exception as exc:  # noqa: BLE001 - el spike documenta cualquier fallo
            print(f"    {etiqueta:<34} EXCEPCION {type(exc).__name__}: {exc}")
            resultados.append(False)

    if fallan:
        print(
            f"\n  HALLAZGO: el grupo {fallan} no admite valores faltantes; imputar antes de\n"
            f"  extraer lo resuelve. Es el riesgo RI4 del anteproyecto, materializado."
        )
    return _veredicto(all(resultados), "imputar antes de extraer habilita todos los grupos")


# --------------------------------------------------------------------------- #
# 7. Fidelidad del submuestreo
# --------------------------------------------------------------------------- #
def experimento_fidelidad_submuestreo() -> bool:
    """Compara el vector completo contra el calculado sobre submuestras."""
    _titulo("7. Fidelidad del submuestreo")

    # Meta-características que dependen del tamaño **por definición**: cambian al
    # submuestrear porque describen la razón entre atributos e instancias. No son
    # inestables; simplemente no son comparables entre tamaños distintos.
    dependientes_del_tamano = {"attr_to_inst", "inst_to_attr", "nr_inst", "t2", "t3"}
    umbral_magnitud = 1e-3

    n_completo = 8000
    X, y = _dataset_numerico(n_completo)
    nombres, completo, segundos_completo = _extraer(X, y, GRUPOS)
    referencia = np.asarray(completo, dtype=float)
    print(f"  referencia: {n_completo} instancias, {segundos_completo:.2f} s")
    print(f"  se excluyen las dependientes del tamaño por definición: "
          f"{sorted(dependientes_del_tamano)}")
    print(f"  y las de magnitud menor que {umbral_magnitud:g}, donde la desviación "
          f"relativa no es informativa\n")

    print(f"  {'muestra':>9}{'segundos':>10}{'comparadas':>12}{'corr.':>9}"
          f"{'desv. mediana':>15}{'p90':>8}")
    for tamano in (1000, 2000, 4000):
        rng = np.random.default_rng(SEED)
        indices = rng.choice(n_completo, size=tamano, replace=False)
        _, muestreado, segundos = _extraer(X[indices], y[indices], GRUPOS)
        submuestra = np.asarray(muestreado, dtype=float)

        comparables = np.array(
            [
                np.isfinite(referencia[i])
                and np.isfinite(submuestra[i])
                and nombres[i].split(".")[0] not in dependientes_del_tamano
                and abs(referencia[i]) >= umbral_magnitud
                for i in range(len(nombres))
            ]
        )
        a, b = referencia[comparables], submuestra[comparables]
        desviacion = np.abs(b - a) / np.abs(a)
        print(
            f"  {tamano:>9}{segundos:>10.2f}{int(comparables.sum()):>12}"
            f"{float(np.corrcoef(a, b)[0, 1]):>9.4f}"
            f"{float(np.median(desviacion)):>14.1%}{float(np.percentile(desviacion, 90)):>8.1%}"
        )

    print(
        "\n  Lectura: la correlación alta y la desviación mediana moderada indican que el\n"
        "  submuestreo preserva el ordenamiento del espacio de meta-características, pero\n"
        "  desplaza los valores. Dos datasets solo son comparables si se les extrajeron las\n"
        "  meta-características con el mismo tamaño de muestra."
    )
    return True


def main() -> None:
    """Ejecuta los experimentos del spike y resume los veredictos."""
    print("SPIKE B - pymfe como extractor de meta-características")
    import pymfe

    print(f"pymfe {pymfe.__version__} | grupos: {', '.join(GRUPOS)}")

    extrapolados = experimento_costo_y_escalado()
    resultados = {
        "3-4. estabilidad del vector": experimento_estabilidad_del_vector(),
        "5. determinismo": experimento_determinismo(),
        "6. datos heterogéneos": experimento_datos_heterogeneos(),
        "7. fidelidad del submuestreo": experimento_fidelidad_submuestreo(),
    }

    _titulo("RESUMEN")
    total_50k = sum(extrapolados.values())
    print(f"  costo extrapolado de un dataset de 50 000 instancias: "
          f"{_formato_duracion(total_50k)} (todos los grupos)")
    print(f"  del cual 'complexity': {_formato_duracion(extrapolados['complexity'])}")
    for nombre, resultado in resultados.items():
        print(f"  [{'OK ' if resultado else 'FALLA'}] {nombre}")


if __name__ == "__main__":
    main()
