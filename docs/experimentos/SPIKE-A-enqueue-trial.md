# Spike A — Viabilidad de `optuna.enqueue_trial` para el arranque en caliente

- **Fecha:** 2026-07-23
- **Etapa:** E1 (análisis y diseño), habilita RE5
- **Entorno:** Optuna 4.9.0, NumPy 2.5.1, scikit-learn 1.9.0, Python 3.12.7
- **Script:** `spikes/spike_a_enqueue_trial.py` (determinista, sin datos externos)
- **Reproducir:** `.\.venv\Scripts\python.exe spikes\spike_a_enqueue_trial.py`

## Por qué se hizo

El pre-registro v0.1 y el diseño del espacio de búsqueda canónico se apoyan en
supuestos sobre el comportamiento de `enqueue_trial` que hasta ahora provenían de
**leer** el código de `OptunaSearchEngine`, no de ejecutarlo. Este spike los verifica
empíricamente antes de que se construya nada encima.

## Resultados

### 1–2. Las inyectadas se consumen primero y cuentan al presupuesto — **confirmado**

Con cinco configuraciones inyectadas y un presupuesto de diez ensayos, los cinco
primeros ensayos del estudio son exactamente las cinco configuraciones inyectadas, en
el orden de inyección, y el estudio ejecuta **diez** ensayos en total, no quince.

Esto confirma empíricamente la decisión E del pre-registro: los cinco ensayos
inyectados consumen presupuesto y deben contarse dentro de «ensayos hasta el
objetivo».

### 3. Reproducibilidad — **confirmado**

Dos corridas con la misma semilla y las mismas inyecciones producen secuencias de
ensayos idénticas. Inyectar cambia la secuencia respecto de no inyectar, de modo que
el efecto es real y no cosmético.

### 4. Configuraciones inválidas — **comportamiento documentado**

| Caso | Comportamiento observado |
|---|---|
| Índice fuera de rango (`paso_a=99`, válidos `0..3`) | `ValueError: '99' not in (0, 1, 2, 3)`. El ensayo **falla** y la excepción se propaga: **aborta el estudio completo** |
| Clave inexistente (`paso_z`) | **Aceptada en silencio.** La clave desconocida se ignora, las válidas se aplican, el ensayo completa sin aviso |
| Configuración parcial (solo `paso_a`) | Aceptada. Los parámetros no especificados se muestrean normalmente |
| Entero de NumPy (`np.int64(2)`) | Aceptado, sin aviso |

Los dos primeros casos son los peligrosos y en direcciones opuestas: **un índice
inválido tumba la búsqueda entera**, y **una clave renombrada se ignora sin dejar
rastro**, aplicando una recomendación parcial que nadie pidió.

### 5. Los parámetros son índices, no transformaciones — **confirmado**

La misma configuración `{"suavizado": 0}` produce `Pipeline([GaussianBlur(kernel_size=3)])`
con un orden de opciones y `Pipeline([MedianFilter(kernel_size=3)])` con otro. El
índice no significa nada por sí mismo. Es la justificación del espacio de búsqueda
canónico y versionado registrado en el pre-registro.

### 6. El `MedianPruner` es inerte — **confirmado**

Con veinte ensayos y un objetivo que no reporta valores intermedios, ningún ensayo
queda en estado `PRUNED`. Los treinta ensayos del protocolo serán treinta
evaluaciones completas, y no hace falta una regla para ensayos podados.

### 7. Frío contra caliente — **hallazgo principal**

Cinco brazos, diez semillas, presupuesto de treinta ensayos, objetivo fijado en el
95 % del mejor valor del frío y censura en 31. Se probaron dos paisajes: uno
**picudo**, donde solo la combinación exacta alcanza la meta, y uno **suave**, con
crédito parcial por cercanía, más parecido a una métrica real como macro-F1.

Mediana de ensayos hasta el objetivo (censurados entre paréntesis):

| Brazo | Paisaje picudo | Paisaje suave |
|---|---|---|
| Frío | **12,0** (0) | **16,0** (0) |
| Caliente, vecinas incorrectas | 31,0 (7) | 31,0 (6) |
| Caliente, recomendador aleatorio | 12,5 (1) | 13,0 (1) |
| Caliente, con el óptimo incluido | 1,0 (0) | 1,0 (0) |
| Caliente, vecinas, `n_startup_trials=5` | 24,0 (1) | 16,0 (0) |

Las «vecinas incorrectas» son cinco configuraciones que coinciden con el óptimo en dos
de tres componentes y ninguna es óptima: simulan un recomendador que acerca pero no
acierta.

**El mecanismo funciona**: con recomendaciones buenas, el arranque en caliente alcanza
el objetivo en el primer ensayo frente a 12 y 16 del frío.

**Pero recomendaciones plausibles y equivocadas son peores que no recomendar nada, y
peores que recomendar al azar.** El efecto aparece en los dos paisajes, así que no es
un artefacto de lo picudo del objetivo. La explicación es el mecanismo de TPE: cinco
puntos agrupados con puntajes altos forman una región que el modelo interpreta como
buena, y la búsqueda se concentra ahí en lugar de explorar. El recomendador aleatorio
no cae en la trampa porque sus puntos están dispersos y puntúan bajo.

Reducir `n_startup_trials` de 10 a 5 mitiga el daño de forma apreciable: en el paisaje
suave devuelve el caliente a la paridad con el frío (16,0 contra 16,0).

## Implicaciones para el proyecto

1. **Validar las configuraciones antes de inyectarlas es obligatorio, no opcional.**
   Un índice fuera de rango aborta la búsqueda completa. El recomendador (RE5) debe
   verificar cada configuración contra la versión vigente del espacio canónico y
   rechazar las que no encajen, en lugar de pasárselas a Optuna.

2. **Una clave desconocida no produce ningún error.** Refuerza el versionado del
   espacio: una recomendación proveniente de otra versión debe rechazarse
   explícitamente, porque Optuna no va a avisar.

3. **`n_startup_trials` es un grado de libertad no declarado que interactúa
   fuertemente con el arranque en caliente.** Hoy `OptunaSearchEngine` lo deja en el
   valor por defecto de TPE (10), de forma implícita.

   **Decisión tomada (2026-07-23):** se fija en **10**, explícitamente, como
   configuración primaria, y se declara un **análisis de sensibilidad a 5**. El mismo
   valor rige en ambas condiciones y en la construcción de la base de
   meta-conocimiento. Razones:

   - El valor primario es el de la librería y no una elección del autor, de modo que
     no puede atribuirse a un ajuste orientado a favorecer la hipótesis. Elegir 5
     —que es el que mejor le sienta al arranque en caliente según este spike— sería
     precisamente seleccionar el parámetro que ayuda al propio resultado.
   - Declarar la sensibilidad de antemano impide elegirla después de ver los
     resultados. Es el mismo patrón que la regla de censura (primario 31, robustez a
     60 y 300), con lo que el pre-registro mantiene un criterio uniforme.
   - Fijarlo de forma explícita en el código, aunque coincida con el valor por
     defecto, protege la reproducibilidad: si una versión futura de Optuna cambia su
     default, los resultados dejarían de ser reproducibles en silencio.

   Consecuencias de implementación: `OptunaSearchEngine` debe exponer
   `n_startup_trials` como parámetro con valor por defecto 10 y pasarlo explícitamente
   a `TPESampler`, y el pre-registro v0.2 debe recoger la decisión y su análisis de
   sensibilidad antes de que se construya la base de meta-conocimiento.

   Advertencia metodológica: el spike midió el efecto de `n_startup_trials` **solo en
   el brazo caliente**. Antes de interpretar la sensibilidad en el experimento real
   habrá que medirlo también en el frío, para distinguir un efecto propio del arranque
   en caliente de un efecto general del muestreador con presupuestos pequeños.

4. **La ablación con recomendador aleatorio no es un trámite.** El pre-registro ya la
   declara como análisis exploratorio para la semana 11; este spike muestra que el
   brazo aleatorio puede *ganarle* a un meta-recomendador mediocre, así que es la
   comparación que dice si el meta-aprendizaje aporta algo o solo aporta ruido.

5. **Consideración de diseño para el recomendador (semanas 7–8):** el modo de fallo es
   un grupo de cinco recomendaciones parecidas entre sí y todas equivocadas. Conviene
   evaluar si conviene priorizar la **diversidad** entre las cinco configuraciones
   recomendadas, y no solo la cercanía en el espacio de meta-características. No es una
   decisión que se tome ahora.

Ninguna de estas implicaciones modifica la Sección 4.3 del anteproyecto.

## Limitaciones de este spike

- El objetivo es sintético y determinista, sobre un espacio de 60 combinaciones. No
  reproduce la geometría de una métrica real sobre un dataset real.
- Las «vecinas incorrectas» se construyeron de forma deliberadamente adversa: todas
  aciertan dos de tres componentes y ninguna es óptima. Un recomendador por vecinos
  reales acertaría el óptimo en algunos datasets.
- El resultado **no** demuestra que el arranque en caliente vaya a fracasar en el
  experimento del proyecto. Demuestra que su beneficio depende críticamente de la
  calidad de las recomendaciones, y que un recomendador mediocre puede ser peor que no
  tener ninguno. Es el riesgo RI1 hecho medible.
