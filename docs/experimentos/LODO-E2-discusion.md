# Discusión de los resultados del experimento 2 (RE6)

> Documento **escrito a mano**. Los números vienen todos de `LODO-E2-resultados.md`,
> que sí se genera; aquí se interpretan.
> Fecha: 2026-09-19. Ejecución analizada: `LODO-E2-resultados.json`, 133,1 min,
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

### La explicación por diversidad queda descartada

> Lo que sigue son **análisis exploratorios**, no confirmatorios, tal como el
> pre-registro v2.0 §8.6 los admite. Se calculan sobre datos ya ejecutados y sobre las
> rejillas exhaustivas ya cacheadas, sin ningún ensayo nuevo. No sostienen ninguna
> afirmación sobre la hipótesis: sirven para explicar el resultado, no para decidirlo.
> Los produce `scripts/analyze_e2_mechanism.py` y quedan en `MECANISMO-E2.md`.

El spike A había propuesto una explicación que parecía natural: cinco configuraciones
parecidas entre sí formarían, para el muestreador, una región falsamente prometedora, y
la similitud reduciría la diversidad de lo inyectado. **Medida sobre este experimento,
esa explicación es falsa.**

| Estrategia | Hamming medio entre las 5 inyectadas | Regret logrado |
|---|---:|---:|
| `nearest` | **2,58** | 0,0058 |
| `hybrid` | 2,54 | 0,0042 |
| `random` | 2,34 | 0,0028 |
| `frequent` | **2,10** | 0,0039 |

`nearest` inyecta los conjuntos **más diversos** de las cuatro y es el que peor
rinde; `frequent` inyecta los **menos diversos** y queda segundo. La diversidad de lo
inyectado no ordena los resultados, y si algo hace es ordenarlos al revés.

### Lo que sí los ordena: la calidad de lo inyectado

La pregunta correcta no es cuán distintas son entre sí las cinco configuraciones, sino
**qué tan buenas son sobre el conjunto excluido**. Eso se puede medir exactamente,
porque la rejilla exhaustiva de cada pareja ya está calculada:

| Estrategia | Regret medio de las 5 | Regret de la **mejor** de las 5 | Inevaluables | Regret logrado |
|---|---:|---:|---:|---:|
| `frequent` | 0,0885 | **0,0150** | 0,0 % | 0,0039 |
| `random` | 0,0819 | 0,0209 | 0,0 % | **0,0028** |
| `hybrid` | 0,0966 | 0,0231 | 2,4 % | 0,0042 |
| `nearest` | 0,0928 | **0,0352** | **2,9 %** | 0,0058 |

La columna que manda es la de la **mejor inyectada**, y su orden reproduce casi
exactamente el del resultado final. **La similitud no identifica configuraciones
mejores para el conjunto consultado: identifica peores.** La mejor de las cinco que
elige `nearest` queda 2,3 veces más lejos del óptimo que la mejor de las cinco que
elige `frequent`, sin mirar nada.

Y hay un costo adicional que se ve en la última columna: **un 2,9 % de lo que `nearest`
inyecta es inevaluable** sobre el conjunto excluido —un ensayo inyectado que se pierde
sin producir métrica—, contra el **0 %** de `frequent` y `random`. Es el problema de
transferibilidad que el desempate canónico ya había anticipado: una ganadora que omite
la codificación no cuesta nada en un conjunto sin categóricas y es inejecutable en uno
que sí las tenga. Elegir por parecido hace más probable arrastrar esas omisiones.

### La lectura que los datos sostienen

Hay una pista más en la misma dirección. A lo largo de los 42 pliegues, `frequent`
inyecta en total **6 configuraciones distintas** y `random` **10**, mientras que
`nearest` inyecta **28**: converge en un puñado de configuraciones buenas en general,
frente a una elección específica por conjunto.

Con todo eso, la explicación que los datos sostienen no es sobre diversidad sino sobre
**generalidad**:

> El arranque en caliente ayuda cuando lo inyectado es **bueno en general**
> (`frequent`, `random`, que muestrean un fondo pequeño de configuraciones ganadoras
> del corpus). `nearest` **no lo es**: apuesta por una señal de similitud que no
> transfiere, y al hacerlo concentra el presupuesto en configuraciones peores para el
> conjunto que tiene delante —y a veces inejecutables—.

Eso explica también por qué `hybrid` no rescata nada: mezclar dos configuraciones por
frecuencia con tres por cercanía queda en 0,0042, peor que `frequent` sola (−7,1 %,
$p = 0{,}96$). Si las tres por cercanía son peores, dosificarlas no ayuda.

**Queda confirmado el efecto de RI1 y refutada la explicación que el spike A le había
dado.** Una recomendación plausible pero equivocada sigue saliendo más cara que no
recomendar nada; lo que no es cierto es que la causa fuera la falta de dispersión.

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
- Nada **causal** a partir de los análisis de la Sección 3. Miden que la similitud
  elige configuraciones peores para el conjunto consultado, lo cual es coherente con
  el resultado y descarta la explicación por diversidad, pero son exploratorios y
  post-hoc: describen el patrón, no lo demuestran.

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
espacio lo que aporta el beneficio es *arrancar con configuraciones buenas en general*,
no *arrancar con las de los conjuntos parecidos*. La similitud, medida sobre las
rejillas exhaustivas, elige configuraciones **peores** para el conjunto que tiene
delante.

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Curvas completas de las 2 100 corridas | `LODO-E2-resultados.json` |
| Tablas y veredictos generados | `LODO-E2-resultados.md` |
| Diversidad y calidad de lo inyectado (exploratorio) | `MECANISMO-E2.md` |
| Óptimos de referencia y meseta al 99 % | `REFERENCIA-E2.json` |
| Protocolo que gobierna esto | `Preregistro-E2.tex` v2.0, etiqueta `prereg-v2.0` |
| La decisión de alcance y su aprobación | `docs/decisiones/ADR-004-segundo-experimento.md` |
| El experimento 1, intacto | `LODO-resultados.md`, `LODO-discusion.md`, `prereg-v1.0` |
