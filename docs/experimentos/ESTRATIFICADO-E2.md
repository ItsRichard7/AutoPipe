# Análisis estratificado por sensibilidad al preprocesamiento

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\analyze_stratified.py`.

> **Análisis secundario declarado**, no exploratorio. El umbral y la estratificación
> los fija el pre-registro v1.0, antes de observar ninguna distribución, y el
> pre-registro del experimento 2 los hereda. No lleva corrección por multiplicidad
> porque no decide sobre la hipótesis: eso lo hacen los dos contrastes confirmatorios.

- **Generado:** 2026-09-09
- **Umbral:** IQR de macro-F1 $\geq$ 0.02
- **Reparto:** 34 sensibles, 8 insensibles

## Por qué ahora tiene contenido

En el **experimento 1** este análisis quedó **vacío por construcción**: el cribado había
seleccionado 25 conjuntos que superaban todos el umbral, de modo que el subconjunto
sensible coincidía con el corpus entero. El pre-registro lo había anticipado y pedía
reportarlo como tal.

El **experimento 2** amplía el corpus a 42 incorporando conjuntos que habían quedado por
debajo del corte por sensibilidad, y **ocho de ellos caen bajo el umbral**. La
estratificación pasa a ser calculable sin haber tocado el umbral, que sigue siendo el
mismo número fijado antes de ver ningún dato.

### Conjuntos del estrato insensible

- `PhishingWebsites` (IQR 0.0027)
- `analcatdata_authorship` (IQR 0.0154)
- `breast-w` (IQR 0.0049)
- `first-order-theorem-proving` (IQR 0.0173)
- `kr-vs-kp` (IQR 0.0110)
- `ozone-level-8hr` (IQR 0.0143)
- `segment` (IQR 0.0072)
- `spambase` (IQR 0.0152)

## Estrato sensible (34 conjuntos)

| Contraste | Regret referencia | Regret condición | Reducción | p (1 cola) | Efecto |
|---|---:|---:|---:|---:|---:|
| `nearest` vs `frio` | 0.0131 | 0.0070 | 46.8 % | 0.0026 | -0.539 |
| `nearest` vs `frequent` | 0.0047 | 0.0070 | -48.7 % | 0.9923 | 0.469 |

## Estrato insensible (8 conjuntos)

| Contraste | Regret referencia | Regret condición | Reducción | p (1 cola) | Efecto |
|---|---:|---:|---:|---:|---:|
| `nearest` vs `frio` | 0.0018 | 0.0010 | 45.9 % | 0.0742 | -0.611 |
| `nearest` vs `frequent` | 0.0012 | 0.0010 | 18.7 % | 0.0391 | -0.722 |

## Cómo leer esto

El modelo base se eligió **sensible al preprocesamiento** precisamente para evitar un
efecto piso: si la métrica apenas variara entre pipelines, ambas condiciones alcanzarían
el objetivo de inmediato y la comparación quedaría vacía. El estrato insensible es, por
definición, el terreno donde ese riesgo es mayor, y por eso el pre-registro pidió
mirarlos por separado.

Los contrastes de este documento **no sustituyen** a los confirmatorios: el estrato
insensible tiene 8 pares y cualquier valor $p$ calculado sobre él
tiene poca potencia. Se reportan por lo que la comparación entre estratos muestra sobre
las condiciones en las que el arranque en caliente ayuda, no como una decisión sobre la
hipótesis.
