# Discusión de los resultados del experimento 2 (RE6)

> Documento **escrito a mano**. Los números vienen todos de `LODO-E2-resultados.md`,
> que sí se genera; aquí se interpretan.
> Fecha: 2026-09-19. Ejecución analizada: `LODO-E2-resultados.json`, 133,1 min,
> 42 conjuntos × 10 semillas × 5 condiciones × 60 ensayos = 126 000 ensayos.
> Actualizado el 2026-09-26 con las secciones 4 bis, 5 ter y 5 quater: la segunda variable
> del anteproyecto, la relación entre meseta y beneficio y la sensibilidad a
> `n_startup_trials`.

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

## 4 bis. La otra variable del anteproyecto tampoco discrimina

El anteproyecto define una segunda variable —el mejor valor alcanzado con un presupuesto
reducido— y el pre-registro v2.0 la declaró secundaria a 5, 15 y 30 ensayos. Estaba guardada
en cada corrida; `PRESUPUESTO-REDUCIDO.md` la reporta sobre los mismos dos pares que los
confirmatorios. Mayor es mejor:

| Ensayos | `nearest` vs `frio` | `nearest` vs `frequent` |
|---:|---|---|
| 5 | gana 26, pierde 14, p = 0,1106 | gana 13, pierde 23, p = 0,9849 |
| 15 | gana 19, pierde 14, p = 0,0945 | gana 21, pierde 12, p = 0,2133 |
| 30 | gana 13, pierde 13, p = 0,3563 | gana 13, pierde 15, p = 0,5272 |

Ninguno es significativo, y no contradice al regret: lo complementa. El mejor valor a un
presupuesto fijo mira **un punto** de la curva, y a esa altura las condiciones están a
milésimas de macro-F1; el regret integra la distancia al óptimo **a lo largo de toda la
curva**, que es donde está el costo. Los datos son compatibles con que el beneficio del
arranque en caliente consista en llegar antes a valores buenos más que en terminar con uno
mejor.

Una advertencia de lectura: a 5 ensayos la mediana de `nearest` (0,7568) queda por debajo de
la del frío (0,7692) y aun así gana en 26 de 42 conjuntos. Las medianas por condición no
describen las diferencias dentro de cada conjunto; el contraste pareado sí.

A igual presupuesto que el experimento 1 —30 ensayos, sobre los 25 conjuntos y las 5 semillas
compartidos— el frío coincide en las **125 de 125** parejas, lo que confirma que los dos
experimentos son comparables. Con la base de 41 conjuntos, `random` sube en 10 conjuntos y baja
en 4; `nearest` sube en 3 y baja en 6.

## 5. La covariable declarada sigue explicando el terreno

La correlación de Pearson entre la anchura de la meseta al 99 % y el regret del arranque en
frío es de **−0,533** (Spearman, −0,791): cuanto más estrecha la meseta, más le cuesta al frío. La meseta
mediana bajó de 36 configuraciones de 144 al 95 % a **14 de 144** al 99 %, con un
mínimo de 2.

Los conjuntos donde el arranque en caliente más ayuda son los de meseta estrecha
—`car` (12 de 144): el frío queda en 0,0564 y `nearest` en 0,0016—, y donde `nearest`
más pierde contra `frequent` son `wilt` (+0,0719), `balance-scale` (+0,0228) y
`dresses-sales` (+0,0209).

## 5 bis. El análisis estratificado, que en el experimento 1 estaba vacío

El pre-registro v1.0 fijó **antes de ver ningún dato** que un conjunto es sensible al
preprocesamiento si su IQR de macro-F1 alcanza 0,02, y declaró una estratificación por
ese umbral. En el experimento 1 quedó **vacía por construcción**: los 25 seleccionados
lo superaban todos. Al ampliar el corpus a 42, **ocho conjuntos caen por debajo** y la
estratificación pasa a ser calculable **sin haber tocado el umbral**.

| Estrato | Contraste | Regret ref. | Regret cond. | Reducción | p (1 cola) |
|---|---|---:|---:|---:|---:|
| Sensibles (34) | `nearest` vs `frio` | 0,0131 | 0,0070 | 46,6 % | **0,0026** |
| Sensibles (34) | `nearest` vs `frequent` | 0,0047 | 0,0070 | −48,9 % | 0,9923 |
| Insensibles (8) | `nearest` vs `frio` | 0,0018 | 0,0010 | 44,4 % | 0,0742 |
| Insensibles (8) | `nearest` vs `frequent` | 0,0012 | 0,0010 | 16,7 % | 0,0391 |

Dos lecturas, y la segunda hay que darla con cuidado.

**La primera es clara: el estrato insensible tiene un orden de magnitud menos margen.**
El arranque en frío deja un regret de 0,0131 en los sensibles y de 0,0018 en los
insensibles. Es exactamente lo que el pre-registro anticipaba al exigir un modelo base
sensible al preprocesamiento: donde el preprocesamiento no cambia la métrica, no hay
costo que ahorrar y la comparación tiende a vaciarse.

**La segunda es que en el estrato insensible `nearest` sí supera a `frequent`**
($p = 0{,}0391$), invirtiendo el resultado global. **No conviene apoyarse en eso**, por
tres razones que se ven en la propia tabla: son **8 pares**, es un análisis secundario
**sin corrección por multiplicidad**, y sobre todo la diferencia es de 0,0012 contra
0,0010 —**dos diezmilésimas de regret**—, una magnitud sin consecuencia práctica. Un
valor $p$ pequeño sobre una diferencia irrelevante es justo lo que una muestra chica
puede producir por azar.

Lo que sí se puede decir es que **la desventaja de la similitud se concentra donde el
preprocesamiento importa**, que es donde el método tendría que servir: en los 34
sensibles, `frequent` le saca a `nearest` un 48,9 % de regret con $p = 0{,}9923$ en la
dirección contraria a la hipótesis.

El detalle está en `ESTRATIFICADO-E2.md`, generado por `scripts/analyze_stratified.py`.

## 5 ter. La meseta y el beneficio

> **Exploratorio**, declarado así en el pre-registro v2.0. Lo produce
> `scripts/tabulate_by_dataset.py` y queda en `LODO-E2-por-dataset.md`. Valores p de dos
> colas y sin corrección.

La sección 5 relaciona la meseta con lo que le cuesta al frío. El pre-registro pedía además
su relación con el **beneficio**: el regret mediano del frío menos el de cada condición.

| Condición | Spearman | p | Pearson | p |
|---|---:|---:|---:|---:|
| `nearest` | −0,400 | 0,0086 | −0,229 | 0,1442 |
| `frequent` | −0,466 | 0,0019 | −0,412 | 0,0067 |
| `random` | −0,546 | 0,0002 | −0,458 | 0,0023 |
| `hybrid` | −0,386 | 0,0116 | −0,359 | 0,0195 |

En las cuatro condiciones **el beneficio crece donde la meseta es estrecha**. Con Pearson, la
de `nearest` deja de ser significativa: la relación se sostiene en rangos, pero no
linealmente. Es la lectura del efecto piso del experimento 1, ahora sobre el beneficio y no
solo sobre la dificultad del frío.

## 5 quater. La conclusión no depende de `n_startup_trials`

El spike A fijó `n_startup_trials` en 10 —el valor por defecto de TPE— y declaró una
sensibilidad a 5, advirtiendo que había que medirla también en el frío.
`scripts/run_sensitivity_startup.py` la ejecutó sobre `frio`, `nearest` y `frequent`: 75 600
ensayos en 90 minutos, con las mismas semillas, óptimos y configuraciones inyectadas. Como
control, los contrastes con 10 recalculados sobre las corridas pareadas coinciden **cifra por
cifra** con los de la sección 1.

| Contraste | Con 10 | Con 5 |
|---|---|---|
| `nearest` vs `frio` | 35,8 %, p Holm 0,0019 — **cumple** | 46,8 %, p Holm 0,0002 — **cumple** |
| `nearest` vs `frequent` | −49,1 %, p 0,9690 — no | −10,2 %, p 0,8761 — no |

- **Los dos veredictos se mantienen.** Con 5 la reducción frente al frío es mayor, y
  `frequent` sigue siendo mejor que `nearest`, aunque por menos.
- **El frío no se mueve** (0,0090 → 0,0094, p = 0,39 de dos colas), ni `nearest` (0,0058 →
  0,0050, p = 0,82). Solo `frequent` empeora (0,0039 → 0,0045, p = 0,029), un valor sin
  corregir que no resistiría Holm sobre los tres brazos (0,086). No se midió por qué y no se le
  atribuye causa.
- **El beneficio frente al frío no cambia de forma significativa**: `nearest` +0,0021 →
  +0,0025 (p = 0,47), `frequent` +0,0034 → +0,0030 (p = 0,18).
- El frío censura más con 5: 47 corridas contra 31.

El temor del spike A —que el beneficio del arranque en caliente dependiera del parámetro— no
se confirma a escala. Detalle en `SENSIBILIDAD-startup.md`.

## 6. Qué se puede afirmar y qué no

**Se puede afirmar:**

- Que el arranque en caliente reduce significativamente el costo de la búsqueda frente
  al arranque en frío en este espacio y este corpus, con el criterio pre-registrado.
- Que esa conclusión no depende de `n_startup_trials`: se sostiene con 10 y con 5.
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
| Estratificación por sensibilidad (secundario declarado) | `ESTRATIFICADO-E2.md` |
| Mejor valor a 5, 15 y 30 ensayos (secundario declarado) | `PRESUPUESTO-REDUCIDO.md` |
| Tabla por conjunto completa y meseta-beneficio (exploratorio) | `LODO-E2-por-dataset.md` |
| Sensibilidad a `n_startup_trials` | `SENSIBILIDAD-startup.md` y `SENSIBILIDAD-startup.json` |
| Óptimos de referencia y meseta al 99 % | `REFERENCIA-E2.json` |
| Protocolo que gobierna esto | `Preregistro-E2.tex` v2.0, etiqueta `prereg-v2.0` |
| La decisión de alcance y su aprobación | `docs/decisiones/ADR-004-segundo-experimento.md` |
| El experimento 1, intacto | `LODO-resultados.md`, `LODO-discusion.md`, `prereg-v1.0` |
