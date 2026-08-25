# Criterios de inclusión del corpus de meta-conocimiento

- **Fecha:** 2026-08-13
- **Etapa:** E1 cierre, entregable RE4
- **Genera:** `docs/CORPUS-candidatos.md` mediante `scripts/list_corpus_candidates.py`

Este documento fija **cómo** se selecciona el corpus. Los criterios de inclusión son los
del anteproyecto y no se alteran; lo que aquí se decide es el universo del que se parte y
cómo se interpretan los criterios cuando el texto admite más de una lectura.

## Criterios de inclusión (Sección 4.3.1 del anteproyecto)

Entre 15 y 25 datasets públicos de clasificación tabular de OpenML o UCI, con:

1. licencia de uso abierto;
2. entre 500 y 50 000 instancias;
3. no más de 100 características;
4. ausencia de datos de imagen o texto embebidos;
5. en datasets multiclase, un máximo de 10 clases.

## Decisiones de procedimiento

### 1. El universo de candidatos es la suite OpenML-CC18

En lugar de filtrar el repositorio completo de OpenML por tamaño, se parte de la suite
curada **OpenML-CC18** (72 datasets de clasificación).

**Razón.** El repositorio de OpenML contiene miles de datasets con versiones duplicadas,
metadatos incompletos y cargas rotas; filtrarlo por tamaño produciría una lista que habría
que depurar a mano, sin criterio reproducible. La CC18 es una colección curada para
comparar algoritmos de clasificación, con metadatos verificados y una identidad estable, y
es la fuente habitual en la literatura de AutoML. Partir de ella hace la selección
**reproducible y citable**.

No modifica la Sección 4.3.1: la CC18 es un subconjunto de OpenML, y los cinco criterios
se aplican igual sobre él.

### 2. El límite de 100 características cuenta solo las predictoras

OpenML incluye la columna objetivo dentro de `NumberOfFeatures`. Se interpreta «no más de
100 características» como **100 variables predictoras**, descontando el objetivo. Es la
lectura natural y la conservadora, porque admite un dataset menos que la alternativa.

La misma corrección se aplica a `NumberOfSymbolicFeatures` al contar variables
categóricas: en clasificación el objetivo es simbólico y se descontaría de más.

### 3. El criterio de imagen y texto lo resuelve el límite de características

Los datasets de píxeles crudos de la CC18 (MNIST, Fashion-MNIST, CIFAR-10) superan
holgadamente las 100 columnas y quedan excluidos por el criterio 3, sin necesidad de una
regla aparte.

Sí sobreviven datasets cuyas variables son **extracciones numéricas** a partir de
imágenes: `mfeat-fourier`, `mfeat-karhunen`, `mfeat-morphological`, `mfeat-zernike`,
`optdigits`, `pendigits`. No son datos de imagen embebidos —son tablas de características
ya calculadas, que es exactamente el objeto del proyecto— así que se admiten. Plantean, en
cambio, un problema distinto que se trata más abajo.

### 4. La licencia se verifica a nivel de suite, no dataset por dataset

Todos los datasets de la CC18 son de acceso público y la suite existe para uso abierto en
investigación. **No se verificó la licencia individual de los 52 candidatos.** La
comprobación se hará sobre los 15 a 25 finalmente seleccionados, en la semana 5, y quedará
registrada en el pre-registro v0.2.

## Resultado

De los 72 datasets de la CC18 sobreviven **52**. Descartes por criterio, contados de forma
independiente (un dataset puede incumplir más de uno):

| Criterio | Descartados |
|---|---:|
| Instancias fuera de [500, 50 000] | 6 |
| Más de 100 características predictoras | 15 |
| Número de clases fuera de [2, 10] | 5 |

Las fichas están en `docs/CORPUS-candidatos.md`, regenerable con el script.

### Cobertura de las restricciones de diversidad

El pre-registro exige, al elegir el corpus final, cubrir los tres regímenes de tamaño y
contar con al menos cinco datasets con variables categóricas y cinco con valores
faltantes. Los candidatos lo permiten con holgura:

| Restricción | Exigido | Disponible |
|---|---:|---:|
| Régimen de centenas | ≥ 1 | 16 |
| Régimen de millares | ≥ 1 | 29 |
| Régimen de decenas de millares | ≥ 1 | 7 |
| Con variables categóricas | ≥ 5 | 19 |
| Con valores faltantes | ≥ 5 | 9 |

También hay 22 datasets multiclase, suficientes para que macro-F1 no se evalúe solo sobre
problemas binarios.

## Hallazgo: los datasets casi duplicados amenazan la validez del protocolo

La lista contiene **familias de datasets que comparten origen**, y eso rompe un supuesto
del protocolo de validación que no estaba contemplado.

**El caso grave.** `mfeat-fourier`, `mfeat-karhunen`, `mfeat-morphological` y
`mfeat-zernike` son cuatro **representaciones distintas de las mismas 2 000 instancias**:
el mismo conjunto de dígitos manuscritos descrito con cuatro conjuntos de características.
Si el corpus incluyera varias y la validación dejara una fuera, el meta-recomendador
seguiría teniendo en su base de entrenamiento *el mismo dataset* bajo otra representación.
La independencia entre pliegues que supone el protocolo *leave-one-dataset-out* dejaría de
cumplirse, y la ventaja del arranque en caliente saldría inflada por una vía difícil de
detectar en los resultados.

**El caso leve.** `kc1`, `kc2`, `pc1`, `pc3`, `pc4` y `jm1` son seis conjuntos de
predicción de defectos de software del mismo repositorio (NASA MDP), con esquemas de
características muy similares aunque de proyectos distintos. La dependencia es menor pero
del mismo tipo.

### Regla de familia (aprobada el 2026-08-24)

Se añade a la selección de la semana 5 una regla de **como máximo un dataset por familia
de origen**, aplicada después del orden por sensibilidad y antes de las restricciones de
diversidad ya pre-registradas.

No modifica los criterios de inclusión de la Sección 4.3.1: es una regla de **selección**
entre candidatos elegibles, del mismo tipo que el desempate por sensibilidad ya acordado.
Se recogerá en el pre-registro v0.2 antes de ejecutar el cribado.

La regla distingue dos niveles, porque el problema no tiene la misma gravedad en ambos:

**Nivel 1 — mismas instancias.** Es un problema de validez y la regla es obligatoria.

| Familia | Datasets | Por qué |
|---|---|---|
| Multiple Features (dígitos) | `mfeat-fourier`, `mfeat-karhunen`, `mfeat-morphological`, `mfeat-zernike` | Son cuatro descripciones de **las mismas 2 000 instancias** |

**Nivel 2 — misma fuente o mismo esquema.** Las instancias son distintas, pero comparten
origen y estructura. La regla es recomendable como medida de diversidad, no de validez.

| Familia | Datasets | Por qué |
|---|---|---|
| NASA MDP (defectos de software) | `kc1`, `kc2`, `pc1`, `pc3`, `pc4`, `jm1` | Seis proyectos del mismo repositorio, con esquemas de características muy similares |
| Reconocimiento de dígitos manuscritos | `optdigits`, `pendigits`, y el representante elegido de `mfeat-*` | Colecciones distintas del mismo dominio |

La distinción importa: en el nivel 1 el meta-recomendador vería literalmente el dataset
excluido dentro de su base de entrenamiento, y la independencia entre pliegues que supone
el protocolo se rompería. En el nivel 2 solo hay similitud de dominio, que es una
propiedad legítima que el recomendador debería poder aprovechar; limitarla protege la
diversidad del corpus, no su validez.

Efecto sobre los 52 candidatos:

| Regla aplicada | Candidatos restantes |
|---|---:|
| Ninguna | 52 |
| Nivel 1: los cuatro `mfeat-*` aportan uno | 49 |
| \+ nivel 2, NASA MDP: los seis aportan uno | 44 |
| \+ nivel 2, dígitos: los tres aportan uno | 42 |

Con 42 familias distintas queda margen sobrado para elegir los 15 a 25 datasets del
corpus, así que la regla no compite con el tamaño comprometido.

## Qué queda para la semana 5

1. Aplicar el cribado de sensibilidad al preprocesamiento sobre los 52 candidatos, con el
   procedimiento ya pre-registrado.
2. Aplicar la regla de familia, ya aprobada.
3. Aplicar las restricciones de diversidad y seleccionar los 15 a 25 finales.
4. Verificar la licencia individual de los seleccionados.
5. Registrar todo en el pre-registro v0.2, antes de ejecutar el cribado.

## Limitaciones

- La lista refleja el estado de OpenML en la fecha de generación. Los identificadores son
  estables, pero conviene regenerarla antes de la selección definitiva.
- Los metadatos provienen de OpenML y no se han verificado descargando cada dataset. Un
  recuento de clases o de valores faltantes erróneo en el repositorio se propagaría a esta
  lista; la descarga de la semana 5 lo detectaría.
- La CC18 excluye por diseño datasets muy pequeños o muy desbalanceados, de modo que el
  corpus hereda ese sesgo. Es una restricción del universo elegido y conviene declararla en
  el informe.
