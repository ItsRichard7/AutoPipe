# ADR-003 — El agrupamiento queda fuera del alcance

- **Estado:** aceptado
- **Fecha:** 2026-09-03
- **Aprobado por:** Ricardo Antonio Borbón Mena, 2026-09-03
- **Etapa:** punto de control de alcance de la semana 7
- **Afecta a:** el alcance del TFG; ninguna interfaz ni ningún módulo existente
- **Insumo:** anteproyecto §4.2 (extensiones deseables), `docs/Agenda.tex`
  (decisiones abiertas), `docs/decisiones/ADR-002-interfaz-de-tarea.md`

## Contexto

El anteproyecto separa el **alcance mínimo comprometido** —detección de anomalías,
clasificación tabular, meta-características, corpus, recomendador, validación
*leave-one-dataset-out* y documentación— de tres **extensiones deseables**, que solo
entran «si el avance lo permite»: el agrupamiento, ampliar el corpus y un meta-modelo
más sofisticado.

La decisión sobre el agrupamiento estaba fechada de antemano para el punto de control
de la semana 7, que es este documento. El ADR-002 dejó el registro de tareas preparado
para recibir una tercera tarea, de modo que la pregunta no es si la arquitectura lo
admite, sino si conviene gastar en ello las horas que quedan.

## Estado en el momento de decidir

| Dato | Valor |
|---|---|
| Horas registradas | 134 de 336 |
| Horas restantes | 202 |
| Semanas restantes | 6 (semanas 7 a 12) |
| Horas comprometidas en esas semanas | 138 |
| Holgura nominal | 64 h |

El alcance mínimo va al día: RE1 a RE4 completos y RE5 funcionalmente completo al
cierre de esta semana. Por horas, el agrupamiento cabría.

## Decisión

**El agrupamiento queda fuera del alcance del TFG.** No se implementa ni se planifica
para ninguna de las semanas restantes.

Las horas de holgura se reservan, en este orden de prioridad, para:

1. subir de tres a cinco semillas por condición en el *leave-one-dataset-out*, si la
   v1.0 del pre-registro lo confirma;
2. la ablación con recomendador aleatorio de la semana 11, que el spike A elevó de
   trámite a comprobación central;
3. el caso completo de anomalías en imágenes de la semana 11, que es lo que sostiene la
   afirmación de que el framework es agnóstico a la tarea;
4. contingencia, que es la función original de la reserva.

## Razones

### 1. No es «una tarea más»: toca el núcleo, no lo extiende

El registro de tareas del ADR-002 agrupa **cuatro** responsabilidades, y el
agrupamiento no encaja en tres de ellas:

| Responsabilidad de `Task` | En agrupamiento |
|---|---|
| Cálculo de métricas | La silueta no recibe etiquetas verdaderas: cambia la **firma** de la llamada |
| Dirección de la mejora | Se conserva sin problema |
| Normalización de predicciones | No hay predicciones que normalizar: hay asignaciones de grupo sin correspondencia con clases |
| Obtención de puntajes | No hay `predict_proba` ni `decision_function` que consultar |

El protocolo de evaluación entero está construido sobre que hay etiquetas:
`DataSplitter` particiona de forma **estratificada por clase**, `evaluate_pipeline`
llama a `model.fit(X_train, y_train)` y compara predicciones contra `labels[val_idx]`.
Nada de eso significa algo sin etiquetas.

Incorporarlo obligaría a modificar `evaluate_pipeline`, que es la función con más
pruebas escritas para fallar ante regresiones —las cuatro garantías anti-fuga— y la
pieza que el riesgo técnico crítico del proyecto ya obligó a rehacer una vez.

### 2. No aporta nada a la hipótesis

La hipótesis del TFG se decide sobre la comparación entre arranque frío y caliente en
clasificación tabular, con su prueba de Wilcoxon sobre 25 pares. Un tercer tipo de
tarea no añade potencia estadística ni cambia el criterio de éxito de §4.3.4: añadiría
un anexo descriptivo al informe.

### 3. Competiría con las semanas que producen el resultado

Las semanas 10 y 11 son las que ejecutan el experimento y analizan sus causas, y son
las de mayor riesgo del cronograma: es donde un resultado inesperado obliga a
reprocesar. La semana 4 ya se pasó de presupuesto (34 h frente a 25), de modo que la
holgura no es tan holgada como su cifra nominal sugiere.

### 4. El corpus tendría que construirse desde cero

Los 25 datasets del corpus se seleccionaron con un criterio —el IQR de macro-F1 sobre
doce pipelines— que no se traslada al agrupamiento, y el propio cribado es un
procedimiento pre-registrado. Un corpus de agrupamiento exigiría su propio criterio de
sensibilidad, su propia base de meta-conocimiento y su propia validación: es
prácticamente el trabajo de las semanas 5 y 6 otra vez.

## Alternativas consideradas

**Aplazar la decisión a la semana 9.** Se descarta: el punto de control de alcance está
fechado en la semana 7 precisamente para que la decisión no se tome cuando ya no quede
margen para actuar sobre ella. Aplazarla mantendría abierta una rama de planificación
sin ningún beneficio, porque los datos que la deciden ya están todos disponibles.

**Implementar solo la métrica de silueta, sin corpus ni experimento.** Se descarta por
ser lo peor de ambas opciones: consume horas del núcleo y produce una capacidad que
nada ejercita, es decir, código muerto. La guía del proyecto lo prohíbe expresamente.

**Sustituir el agrupamiento por otra extensión deseable.** No se decide aquí. Ampliar
el corpus y el meta-modelo más sofisticado siguen siendo extensiones deseables del
anteproyecto y su valoración corresponde a las semanas 10 y 11, cuando se sepa si el
alcance mínimo cerró con margen.

## Consecuencias

- El anteproyecto **no se modifica**: el agrupamiento seguía siendo una extensión
  deseable y condicional, así que descartarlo no reduce ningún compromiso adquirido.
  Lo que este documento hace es cerrar una condición, no cambiar el alcance.
- El pre-registro **no se ve afectado**: su sección «qué queda fuera de este documento»
  ya anticipaba que esta decisión no toca el protocolo registrado.
- Las métricas primarias del anteproyecto se reducen de hecho a dos, macro-F1 para
  clasificación y AUC-ROC para anomalías. La silueta figuraba en §4.3.2 como
  «clustering si entra», y no entra.
- El registro de tareas sigue admitiendo una tercera tarea sin cambios estructurales.
  La decisión es de alcance, no de arquitectura, y no cierra la puerta a un trabajo
  posterior.
- La generalidad del framework se sostiene igual: se demuestra con **dos** tareas de
  modalidades distintas, imágenes y tablas, que es lo que el alcance mínimo
  comprometía.

## Verificación

- El ítem «entrada o descarte del agrupamiento» se retira de las decisiones abiertas de
  `docs/Agenda.tex` y la decisión queda anotada en la bitácora del día.
- No hay código que verificar: este ADR no cambia ninguna línea.
