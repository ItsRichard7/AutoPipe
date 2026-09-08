# Discusión de los resultados del LODO (RE6)

> Documento **escrito a mano**. Los números vienen todos de
> `LODO-resultados.md`, que sí se genera; aquí se interpretan.
> Fecha: 2026-09-15. Ejecución analizada: `LODO-resultados.json`, 13,4 min.

## 1. Qué dice el contraste pre-registrado

**La hipótesis no se confirma.** El criterio del anteproyecto (§4.3.4) exige las dos
condiciones a la vez, y solo se cumple una:

| Condición del criterio | Exigido | Observado | ¿Cumple? |
|---|---:|---:|:--:|
| Reducción de la mediana de *ensayos hasta el objetivo* | ≥ 30 % | 33,3 % | sí |
| Wilcoxon pareada de una cola | p < 0,05 | p = 0,3115 | **no** |

La reducción del 33,3 % merece una advertencia antes de que nadie la cite: es la
mediana de 3 ensayos bajando a 2. **Un solo ensayo.** A estas magnitudes el umbral
relativo del 30 % es casi vacuo —cualquier mejora de un ensayo sobre una mediana de
tres lo supera— y el que de verdad discrimina es el valor p. Esto no es una excusa
construida a posteriori: es una limitación del criterio que solo se puede ver una vez
medida la escala real de la variable, y queda anotada como tal.

El resultado es **estable frente a la censura**: recalcular con castigos de 60 y de
300 en lugar de 31 no cambia ni una cifra. La razón es aritmética y conviene decirla,
porque de otro modo la robustez parece más fuerte de lo que es: solo **1 de las 125**
corridas calientes quedó censurada, y una sola corrida no puede mover la mediana de
cinco semillas. El arranque en frío no censuró ninguna, que es lo que el pre-registro
demuestra que tiene que pasar.

## 2. El hallazgo que importa: dónde se pierde la significancia

El recuento por conjunto de datos parece favorable —el caliente llega antes en 14,
empata en 4 y llega después en 7— y sin embargo la prueba no se acerca a 0,05. La
razón está en las **magnitudes**, que es justo lo que Wilcoxon pondera y un recuento
ignora:

| Estrategia | Gana | Empata | Pierde | Ahorro medio al ganar | Costo medio al perder | Suma neta |
|---|---:|---:|---:|---:|---:|---:|
| `nearest` (primaria) | 14 | 4 | 7 | −2,14 ensayos | **+3,57 ensayos** | −5 |
| `random` (control) | 12 | 4 | 9 | −2,50 | +2,44 | −8 |
| `frequent` (control) | 12 | 9 | 4 | −2,75 | **+2,25** | **−24** |

`nearest` **gana a menudo y poco, y pierde raro y mucho**. Sus cuatro peores casos
—`wilt` +5, `climate-model-simulation-crashes` +5, `balance-scale` +5, `pc1` +4—
cuestan cada uno más que lo que ahorra un acierto típico. Esa asimetría es exactamente
la que el spike A había medido en pequeño y que el anteproyecto registró como riesgo
**RI1**: *una recomendación plausible pero equivocada sale más cara que no recomendar
nada*. El experimento completo la reproduce sobre el corpus entero.

## 3. El control gana a la estrategia primaria

`frequent` —que inyecta las configuraciones ganadoras **más frecuentes del pliegue** y
no mira las meta-características del conjunto consultado— alcanza p = 0,0239 y cumpliría
el criterio. `nearest`, que sí las mira, no.

**Esto no confirma la hipótesis del trabajo**, y el reporte está construido para que no
se pueda leer así: la fila primaria está marcada como tal y el criterio se juega solo
ahí. Lo que dice es algo distinto y más informativo:

> En este espacio y sobre este corpus, **un buen valor por defecto vale más que la
> similitud entre conjuntos de datos**. La capa de meta-aprendizaje no está aportando
> señal por encima de «inyectá lo que suele funcionar».

Es el resultado de una ablación, y llega antes de lo previsto: la ablación con
recomendador aleatorio estaba planificada para la semana 11 y los controles secundarios
del pre-registro ya la contienen. `random` —ganadoras de conjuntos al azar— queda en
p = 0,2693, es decir, **indistinguible de `nearest`**. Los tres números juntos ordenan
así las explicaciones:

- Si la similitud aportara señal, `nearest` debería separarse de `random`. No lo hace.
- Si el beneficio viniera de arrancar con configuraciones *razonables* en vez de
  aleatorias del espacio, `frequent` debería ser la mejor. Lo es.

## 4. Por qué había tan poco margen: el efecto piso

La covariable que se declaró **antes** de ejecutar es la que explica el tamaño del
efecto, y esa es la razón de haberla declarado antes:

- Mediana del arranque en frío: **3 ensayos**. Media 3,36. Máximo 8.
- En **12 de 25** conjuntos el frío ya alcanza el objetivo en **2 ensayos o menos**.
- Correlación de Pearson entre la anchura de la meseta y los ensayos que necesita el
  frío: **−0,668**.

Con el objetivo definido al 95 % del óptimo del frío y mesetas de hasta 108 de 144
configuraciones, la búsqueda en frío tropieza con una configuración suficientemente
buena casi de inmediato. **No hay costo que ahorrar.** Los conjuntos donde el caliente
ayuda de verdad son precisamente los de meseta estrecha —`car` (12/144): de 7 a 1;
`tic-tac-toe` (24/144): de 5 a 1; `credit-approval` (26/144): de 5 a 1— y los tres
peores casos del caliente tienen mesetas anchas, donde equivocarse cuesta sin que
acertar sirviera de mucho.

Esta es la afirmación sustantiva que la covariable permite hacer y que sin ella no se
podría: no es que el arranque en caliente no funcione, es que **el espacio canónico
`tabular-v1` es demasiado fácil para que el ahorro se pueda medir**. Es una afirmación
sobre las condiciones en las que el método puede ayudar, no una excusa.

## 5. Qué no se puede concluir

- **No** se puede concluir que el arranque en caliente meta-aprendido no sirva. Se
  midió en un espacio de 144 pipelines con mesetas anchas y con un presupuesto de 30
  ensayos; en un espacio con más pasos, o con modelos cuyo desempeño dependa más del
  preprocesamiento, el margen sería otro.
- **No** se puede cambiar el criterio, el espacio, el corpus ni el protocolo ahora. La
  regla de oro del pre-registro lo prohíbe y hacerlo invalidaría todo lo anterior.
  Cualquier medición en un espacio más difícil sería un experimento **nuevo**, con su
  propio pre-registro, y no un reanálisis de este.
- **No** se puede presentar el resultado de `frequent` como confirmación. Es un análisis
  secundario declarado, y su lectura correcta es la de la sección 3.

## 6. Consecuencias para lo que queda

1. La ablación de la semana 11 **ya está hecha** en lo esencial, y su resultado es
   informativo: la holgura que reservaba puede ir al análisis de causas y al caso de
   imágenes.
2. El análisis estratificado por sensibilidad al preprocesamiento sigue vacío por
   construcción —el cribado dejó los 25 por encima del umbral—, tal como el
   pre-registro anticipó.
3. Para el informe, el capítulo de resultados tiene un hallazgo con estructura, no un
   «no significativo»: una asimetría medida entre acierto y error, un control que
   supera a la estrategia primaria, y una covariable declarada de antemano que explica
   el tamaño del efecto. Medir con rigor que un método no ayuda **y poder decir por
   qué** es un resultado.

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Curvas completas de las 500 corridas | `LODO-resultados.json` |
| Tablas y veredicto generados | `LODO-resultados.md` |
| Reporte con gráficas | `reportes/lodo.html`, `reportes/lodo.pdf` |
| Protocolo que gobierna todo esto | `Preregistro.tex` v1.0, etiqueta `prereg-v1.0` |
| Anchura de la meseta por conjunto | `DIVERSIDAD-valores.json` |
