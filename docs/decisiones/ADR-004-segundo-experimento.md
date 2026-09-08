# ADR-004 — Un segundo experimento, con su propio pre-registro

- **Estado:** propuesto — **requiere la aprobación del profesor asesor antes de ejecutar**
- **Fecha:** 2026-09-17
- **Propuesto por:** Ricardo Antonio Borbón Mena
- **Etapa:** después de ejecutar el experimento 1 (semana 10), antes de la semana 11
- **Afecta a:** el alcance del TFG. Amplía lo comprometido; no retira nada
- **Insumo:** `docs/experimentos/LODO-resultados.md`, `LODO-discusion.md`, anteproyecto
  §4.2 y §4.3, `docs/experimentos/Preregistro.tex` v1.0

## Contexto

El experimento 1 se ejecutó el 2026-09-13 bajo el pre-registro v1.0 y **la hipótesis no
se confirmó**: reducción del 33,3 % de la mediana —que cumple el umbral del anteproyecto—
con un valor $p$ de una cola de 0,3115, que no lo cumple.

El resultado está reportado y no se toca. Pero su propio análisis dejó al descubierto que
el experimento tenía **poca resolución**, y esa es una afirmación que ahora se puede
sostener con números en vez de con intuición.

### Lo que midió el diagnóstico

Se midió sobre los datos ya ejecutados del experimento 1, **leyendo únicamente la
condición de arranque en frío**. Ninguna curva caliente se consultó.

Endurecer el objetivo, sin ejecutar un solo ensayo nuevo:

| Objetivo (fracción del propio mejor del frío) | Mediana de ensayos | Conjuntos con margen (≥5) | Saturados (≤2) |
|---|---:|---:|---:|
| 95 % (el del anteproyecto) | 3,0 | 7 de 25 | 12 de 25 |
| 97 % | 5,0 | — | 8 de 25 |
| **99 %** | **10,0** | **20 de 25** | 2 de 25 |
| 100 % | 10,0 | — | 1 de 25 |

Y la variable medida:

| Variable | Valores distintos en 25 conjuntos | Colisiones |
|---|---:|---:|
| Ensayos hasta el objetivo (entero, 1–8) | 7 | **18** |
| *Regret* normalizado sobre la curva | 25 | **0** |

La prueba de Wilcoxon **descarta los empates**. Con 18 colisiones de 25, el contraste
primario del experimento 1 operó sobre una fracción de la muestra disponible.

### Qué se concluye de eso

El factor dominante del resultado nulo **no es el tamaño del espacio de búsqueda** sino
la holgura del umbral del 95 % y la granularidad de la variable. Es una conclusión
distinta de la que se anticipó al escribir la discusión, y llegó por medirla.

De paso invierte el juicio sobre el espacio canónico: `tabular-v1` tiene solo 144
configuraciones, y eso resulta ser **un activo**, porque permite calcular el óptimo
exacto por pareja (dataset, semilla) con una rejilla exhaustiva. En un espacio con
hiperparámetros continuos habría que conformarse con un proxy.

## Decisión

**Se ejecuta un segundo experimento, gobernado por un pre-registro propio
(`Preregistro-E2.tex`, v2.0), y se reportan los dos.**

Tres propiedades hacen que esto no sea una modificación retroactiva del primero:

1. **El experimento 1 queda intacto.** Sus números, su criterio y su veredicto se reportan
   tal como salieron. El pre-registro v1.0 no se edita: sigue siendo el documento que lo
   gobernó, con su etiqueta `prereg-v1.0`.
2. **§4.3 ya está cumplido.** El experimento comprometido ante el tribunal —corpus de
   15–25 datasets, 30 ensayos, Wilcoxon pareada, criterio del 30 %— se ejecutó completo.
   Lo que viene es **adición**, nunca sustitución.
3. **El pre-registro v2.0 se cierra y se etiqueta antes de ejecutar**, como los
   anteriores. La regla de oro se mantiene.

### Qué cambia en el experimento 2

| Cambio | De | A | Por qué |
|---|---|---|---|
| Objetivo | 95 % del propio mejor del frío | **99 % del óptimo exhaustivo** por (dataset, semilla) | La palanca medida arriba; y una vara externa e igual para ambas condiciones |
| Presupuesto | 30 ensayos | **60** | Al 99 % la peor corrida del experimento 1 llegó a 30 exactos: con ese objetivo, 30 se queda corto |
| Variable primaria | Ensayos hasta el objetivo | ***Regret* normalizado**; los ensayos pasan a secundaria declarada | Elimina los 18 empates y conserva la comparabilidad con el experimento 1 |
| Corpus | 25 datasets | **42** | De 25 a 42 pares de Wilcoxon; ya están descargados y verificados |
| Semillas | 5 | **10** | El cómputo son minutos |
| Condiciones | 4 | **5**: se añade `hybrid` | Separa la mejora del instrumento de la mejora del método |

### Qué NO cambia, deliberadamente

- **El espacio de búsqueda sigue siendo `tabular-v1`**, sin tocar. Se evaluó construir un
  `tabular-v2` con hiperparámetros continuos y se descartó: el diagnóstico mostró que no
  era la causa dominante, costaba unas 26 h y habría impedido calcular el óptimo exacto.
- **El modelo base, las meta-características, las transformaciones y las cuatro garantías
  del protocolo de evaluación** se heredan sin modificación.
- **La regla de familia se mantiene**, y por eso el corpus crece a 42 y no a 52. De los
  27 candidatos fuera del corpus, diez lo estaban por regla de familia —seis de «dígitos
  manuscritos» y cuatro de «NASA MDP»— y siguen excluidos. Admitirlos sería grave
  justamente en una validación por exclusión de conjunto: en el pliegue que excluye a uno,
  su casi gemelo seguiría entrenando al recomendador, y el vecino más cercano sería un
  hermano. El sesgo iría **a favor** de la hipótesis.
- **`nearest` se mantiene exactamente como en la v1.0.** Es lo que permite atribuir lo que
  cambie: si `nearest` contra frío mejora en el experimento 2, es por el instrumento y no
  por el método, porque el método es el mismo.
- **`frequent` sigue siendo un control.** Ganó en el experimento 1, y retirarlo o
  degradarlo ahora sería seleccionar sobre el resultado.
- **La ponderación de meta-características queda fuera** de las condiciones confirmatorias.
  Toda ponderación exige un criterio de ajuste y no encontré ninguno que se pueda fijar
  *a priori* sin arbitrariedad. Se reporta como análisis exploratorio, si da tiempo.

## Alternativas consideradas

| Alternativa | Por qué se descarta |
|---|---|
| No hacer nada y cerrar con el resultado nulo | Cumple lo comprometido, pero deja sin responder si el nulo es del método o del instrumento, habiendo horas para averiguarlo |
| Volver a ejecutar con el umbral al 99 % y reportar **eso** como resultado | Es el fraude que el pre-registro existe para impedir: cambiar la regla después de ver el resultado |
| Construir `tabular-v2` con hiperparámetros continuos | El diagnóstico lo desmintió como causa dominante; 26 h y el mayor riesgo de cronograma; e impide el óptimo exacto |
| Cambiar solo el recomendador | Volvería a chocar con el efecto piso, que es lo que impide medir |
| Cambiar instrumento y método a la vez, sin `nearest` intacto | Confunde las dos causas: un resultado positivo no se podría atribuir |

## Consecuencias

**Positivas.** El informe pasa de tener un resultado nulo a tener una secuencia:
resultado nulo → diagnóstico cuantificado con una covariable declarada de antemano →
rediseño del instrumento → segundo experimento pre-registrado. Para un TFG eso demuestra
más criterio que un resultado favorable a la primera. Y la comparación `nearest` contra
`frequent` responde la pregunta que el experimento 1 dejó planteada.

**Negativas y riesgos.**

- El alcance crece cuando quedan seis semanas. Mitigación: son ~40 h de las ~115
  disponibles, y el caso de imágenes y el informe quedan con holgura.
- El experimento 2 puede volver a dar nulo. Está previsto y se reporta igual: el
  pre-registro v2.0 declara el resultado negativo como desenlace legítimo, igual que la
  v1.0.
- **Dos declaraciones obligatorias** viajan al pre-registro v2.0, y sin ellas esto no
  sería limpio: que el umbral del 99 % se eligió midiendo sobre datos ya ejecutados
  —solo de la condición fría—, y que 17 de los 42 datasets se usaron en la evaluación
  exploratoria de la semana 8.

## Aprobación

**Esta decisión modifica parámetros que el anteproyecto §4.3 fijó y que están
comprometidos ante el tribunal.** Aunque solo los amplía y no retira nada, requiere el
visto bueno del profesor asesor antes de ejecutar el experimento 2. Hasta entonces el
estado de este ADR es «propuesto».

## Referencias

- `docs/experimentos/Preregistro.tex` v1.0 — el protocolo del experimento 1, inmutable
- `docs/experimentos/Preregistro-E2.tex` v2.0 — el protocolo del experimento 2
- `docs/experimentos/LODO-resultados.md` y `LODO-discusion.md` — lo que se midió y su lectura
- `docs/decisiones/ADR-003-alcance-agrupamiento.md` — la holgura que este ADR gasta
