# Discusión de los resultados del experimento 2 (RE6)

> Documento **escrito a mano**. Los números vienen todos de `LODO-E2-resultados.md`,
> que sí se genera; aquí se interpretan.
> Fecha: 2026-09-21. Ejecución analizada: `LODO-E2-resultados.json`, 133,1 min,
> 42 conjuntos × 10 semillas × 5 condiciones × 60 ensayos = 126 000 ensayos.

## 1. Las dos respuestas, que van en direcciones opuestas

El pre-registro v2.0 declaró **dos** contrastes confirmatorios y corrección de Holm
sobre la familia. Los dos se resolvieron, y hay que leerlos juntos porque cada uno
sin el otro engaña.

| Contraste | Regret referencia | Regret condición | Reducción | p (Holm) | Cumple |
|---|---:|---:|---:|---:|:--:|
| `nearest` vs `frio` | 0,0090 | 0,0058 | **35,8 %** | **0,0019** | **sí** |
| `nearest` vs `frequent` | 0,0039 | 0,0058 | −49,1 % | 0,9690 | no |

**La hipótesis del anteproyecto se confirma.** El arranque en caliente meta-aprendido
reduce el costo de la búsqueda frente al arranque en frío: 35,8 % de reducción del
regret mediano —el criterio exige 30 %— con un valor $p$ ajustado por Holm de 0,0019,
y un tamaño del efecto de −0,537. Gana en 32 de los 42 conjuntos.

**Y el mecanismo que el trabajo postulaba no es el que produce ese beneficio.**
`frequent` —que inyecta las ganadoras más frecuentes del pliegue y **no mira las
meta-características del conjunto consultado**— obtiene un regret de 0,0039 contra los
0,0058 de `nearest`. La similitud no solo no aporta por encima de un buen valor por
defecto: queda **por debajo** de él.

Presentar solo la primera fila sería engañoso; presentar solo la segunda, también.

## 2. La estrategia primaria es la peor de las cuatro calientes

| Condición | Regret mediano | Censuradas | Qué inyecta |
|---|---:|---:|---|
| `random` | **0,0028** | 32 / 420 | Ganadoras de conjuntos **al azar** |
| `frequent` | 0,0039 | 29 / 420 | Las ganadoras más frecuentes del pliegue |
| `hybrid` | 0,0042 | 32 / 420 | 2 por frecuencia + 3 por cercanía |
| `nearest` | 0,0058 | **45 / 420** | Las de los conjuntos **más parecidos** |
| `frio` | 0,0090 | 31 / 420 | Nada |

Las cuatro condiciones calientes baten al arranque en frío. La que el anteproyecto
compromete como método es **la última de las cuatro**, y la ablación aleatoria es la
mejor. Es el resultado más incómodo del trabajo y también el más informativo.

## 3. Por qué pierde: la asimetría, ahora cuantificada

Todas las condiciones calientes ganan en aproximadamente el mismo número de conjuntos.
La diferencia no está en cuántas veces aciertan sino en **cuánto cuesta cuando
fallan**:

| Condición | Gana | Pierde | Ahorro medio al ganar | Costo medio al perder |
|---|---:|---:|---:|---:|
| `nearest` | 32 | 10 | −0,0095 | **+0,0077** |
| `frequent` | 32 | 10 | −0,0133 | +0,0021 |
| `random` | 34 | 8 | −0,0121 | +0,0029 |
| `hybrid` | 31 | 11 | −0,0124 | +0,0025 |

`nearest` **paga 3,7 veces más caro cada error** que `frequent`, y ahorra menos cuando
acierta. Su tasa de censura lo confirma desde otro ángulo: **45 corridas de 420
(10,7 %) no alcanzan el objetivo, más que el propio arranque en frío** (31, 7,4 %).
Recomendar por similitud deja la búsqueda peor que no recomendar nada más a menudo
que cualquier otra condición, incluida la de no recomendar.

Es el riesgo **RI1** del anteproyecto, que el spike A midió en pequeño, reproducido
sobre 42 conjuntos y con la resolución suficiente para separarlo de las alternativas.

**La explicación mecánica es la que el spike A ya había propuesto.** Cinco
configuraciones parecidas entre sí forman, para el muestreador TPE, una región
aparentemente prometedora, y la búsqueda se concentra ahí en lugar de explorar. Cinco
configuraciones dispersas —que es lo que produce elegir conjuntos al azar— le dan al
muestreador un mapa mejor del espacio. **La similitud reduce la diversidad de lo
inyectado, y esa diversidad es justamente lo que el muestreador necesita.**

Eso explica también por qué `hybrid` no rescata nada: mezclar dos configuraciones por
frecuencia con tres por cercanía queda en 0,0042, peor que `frequent` sola (−7,1 %,
$p = 0{,}96$). El problema no se arregla dosificando la similitud.

## 4. La variable fue la mitad del problema, y ahora se puede demostrar

El pre-registro conservó los *ensayos hasta el objetivo* como análisis secundario
precisamente para poder hacer esta comparación. Sobre **exactamente los mismos datos**,
el mismo corpus y el mismo objetivo:

| Variable | Reducción de la mediana | p (una cola) | ¿Discrimina? |
|---|---:|---:|:--:|
| *Regret* normalizado | 35,8 % | **0,0010** | sí |
| Ensayos hasta el objetivo | 41,4 % (7,2 → 4,2) | 0,3902 | **no** |

Con la variable del experimento 1, este experimento **habría vuelto a dar nulo** pese
al objetivo endurecido, al corpus de 42 y a las diez semillas. La reducción de la
mediana es incluso mayor en esa variable —41,4 % contra 35,8 %— y aun así no alcanza
significancia, porque el entero colapsa conjuntos distintos en el mismo valor y
Wilcoxon descarta los pares empatados.

Es la confirmación directa del diagnóstico del ADR-004, y una lección metodológica
que vale para el informe: **una variable de baja resolución puede ocultar un efecto
real de magnitud considerable**, y ninguna cantidad de datos adicionales lo arregla.

## 5. La covariable declarada sigue explicando el terreno

La correlación entre la anchura de la meseta al 99 % y el regret del arranque en frío
es de **−0,533**: cuanto más estrecha la meseta, más le cuesta al frío. La meseta
mediana bajó de 36 configuraciones de 144 al 95 % a **14 de 144** al 99 %, con un
mínimo de 2.

Los conjuntos donde el arranque en caliente más ayuda son los de meseta estrecha
—`car` (12 de 144): el frío queda en 0,0564 y `nearest` en 0,0016—, y donde `nearest`
más pierde contra `frequent` son `wilt` (+0,0719), `balance-scale` (+0,0228) y
`dresses-sales` (+0,0209).

## 6. Qué se puede afirmar y qué no

**Se puede afirmar:**

- Que el arranque en caliente reduce significativamente el costo de la búsqueda frente
  al arranque en frío en este espacio y este corpus, con el criterio pre-registrado.
- Que el beneficio **no proviene de la similitud entre conjuntos de datos**, y que la
  selección por vecinos más cercanos es la peor de las cuatro formas ensayadas de
  arrancar en caliente.
- Que una recomendación por similitud equivocada sale más cara que no recomendar nada,
  con la censura y el costo medio del error para respaldarlo.

**No se puede afirmar:**

- Que el meta-aprendizaje no pueda funcionar. Se midió **una** familia de
  recomendadores —vecinos más cercanos sobre 88 meta-características con distancia L1
  y escalado robusto— fijada *a priori*. Un meta-modelo distinto es una pregunta
  abierta, y el anteproyecto ya la listaba como extensión deseable.
- Que el resultado se extienda a espacios mayores, a otros modelos base o a
  presupuestos distintos de 60 ensayos.
- Nada a partir de los análisis secundarios como si fueran confirmatorios. `random`
  supera a `nearest` de forma llamativa, pero ese contraste no estaba declarado como
  confirmatorio y se reporta como lo que es.

**Y no se puede cambiar nada ahora.** Ni el criterio, ni el espacio, ni el corpus, ni
las condiciones. Los dos pre-registros están congelados y etiquetados, y cualquier
medición nueva sería un experimento nuevo con su propio documento.

## 7. Los dos experimentos juntos

| | Experimento 1 | Experimento 2 |
|---|---|---|
| Vara | 95 % del propio mejor del frío | 99 % del óptimo exhaustivo |
| Variable | Ensayos (7 valores, 18 empates) | Regret (sin empates) |
| Corpus · semillas | 25 · 5 | 42 · 10 |
| `nearest` vs `frio` | no se confirma ($p = 0{,}31$) | **se confirma** ($p_{\text{Holm}} = 0{,}0019$) |
| ¿Aporta la similitud? | no (`frequent` mejor) | **no** (`frequent` mejor, con potencia) |

El experimento 1 no pudo detectar un efecto que sí existía, y su diagnóstico dijo por
qué. El experimento 2 lo detecta y, con la resolución recuperada, responde además la
pregunta que el primero solo pudo insinuar.

**La conclusión que el trabajo sostiene no es «el meta-aprendizaje funciona» ni «no
funciona», sino una más precisa y más útil:** el arranque en caliente ayuda, y en este
espacio lo que aporta el beneficio es *arrancar con configuraciones plausibles y
diversas*, no *arrancar con las de los conjuntos parecidos*.

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Curvas completas de las 2 100 corridas | `LODO-E2-resultados.json` |
| Tablas y veredictos generados | `LODO-E2-resultados.md` |
| Óptimos de referencia y meseta al 99 % | `REFERENCIA-E2.json` |
| Protocolo que gobierna esto | `Preregistro-E2.tex` v2.0, etiqueta `prereg-v2.0` |
| La decisión de alcance y su aprobación | `docs/decisiones/ADR-004-segundo-experimento.md` |
| El experimento 1, intacto | `LODO-resultados.md`, `LODO-discusion.md`, `prereg-v1.0` |
