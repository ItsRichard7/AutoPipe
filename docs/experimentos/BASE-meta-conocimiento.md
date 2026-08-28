# Base de meta-conocimiento (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\build_meta_knowledge.py`.

- **Generado:** 2026-08-28
- **Datasets:** 25
- **Ensayos por dataset:** 50, semilla 42
- **Versión del espacio:** `tabular-v1`
- **Métrica:** macro_f1
- **Datos completos:** `BASE-meta-conocimiento.json`

## Qué es

Lo que el recomendador de arranque en caliente consulta. Dado un dataset nuevo,
busca aquí los más parecidos comparando vectores de meta-características e
inyecta sus configuraciones ganadoras como ensayos iniciales.

## Meta-características

- Extraídas por dataset: **107**
- Sobreviven a la regla de descarte: **88**
- Descartadas por resultar no finitas en algún dataset: **19**

La regla del pre-registro descarta toda meta-característica no finita en algún
dataset del corpus, sin imputar al nivel meta. Las descartadas son:

`can_cor.sd`, `cat_to_num`, `f1v.mean`, `f1v.sd`, `f2.mean`, `f2.sd`, `f3.sd`, `f4.sd`, `g_mean.mean`, `g_mean.sd`, `h_mean.mean`, `h_mean.sd`, `l1.sd`, `l2.sd`, `l3.sd`, `n2.mean`, `n2.sd`, `num_to_cat`, `sd_ratio`

El vector completo se conserva en cada entrada: la regla depende del corpus, así
que guardar solo su resultado la volvería irreversible.

## Diversidad de los pipelines ganadores

**20 configuraciones distintas entre 25 datasets.**

Es la condición para que el meta-aprendizaje tenga sentido. Si todos los datasets
compartieran el mismo ganador, el recomendador no tendría nada que aprender:
bastaría recomendar siempre esa configuración, y la comparación entre arranque
frío y caliente mediría otra cosa.

## Ensayos inevaluables

96 de 1250 ensayos (7.7 %) resultaron
inevaluables. Es muy inferior a la fracción de la región inevaluable del espacio,
que llega al 56 % en datasets con categóricas y faltantes: el muestreador aprende
a esquivarla después de los ensayos de arranque.

## Entradas

La configuración es el índice elegido por paso; el índice mayor de cada paso
significa omitirlo, y solo es interpretable con la versión del espacio indicada
arriba.

| OpenML ID | Nombre | macro_f1 | Fallidos | Segundos | Configuración ganadora |
|---:|---|---:|---:|---:|---|
| 40983 | `wilt` | 0.8859 | 0 | 2.6 | `encoding=2 imputation=1 scaling=2 selection=3` |
| 38 | `sick` | 0.9171 | 15 | 2.7 | `encoding=0 imputation=1 scaling=2 selection=1` |
| 18 | `mfeat-morphological` | 0.7409 | 0 | 1.0 | `encoding=1 imputation=0 scaling=1 selection=2` |
| 40975 | `car` | 0.9108 | 8 | 1.1 | `encoding=0 imputation=0 scaling=1 selection=2` |
| 50 | `tic-tac-toe` | 0.9648 | 8 | 1.2 | `encoding=0 imputation=0 scaling=2 selection=2` |
| 40982 | `steel-plates-fault` | 0.7725 | 0 | 2.1 | `encoding=1 imputation=1 scaling=2 selection=3` |
| 188 | `eucalyptus` | 0.6195 | 14 | 2.1 | `encoding=1 imputation=1 scaling=2 selection=0` |
| 29 | `credit-approval` | 0.8536 | 13 | 1.4 | `encoding=0 imputation=0 scaling=0 selection=0` |
| 11 | `balance-scale` | 0.5906 | 0 | 0.6 | `encoding=2 imputation=1 scaling=3 selection=3` |
| 1068 | `pc1` | 0.7669 | 0 | 1.5 | `encoding=1 imputation=2 scaling=2 selection=0` |
| 41027 | `jungle_chess_2pcs_raw_endgame_complete` | 0.6995 | 0 | 11.3 | `encoding=2 imputation=1 scaling=3 selection=2` |
| 1464 | `blood-transfusion-service-center` | 0.6637 | 0 | 0.7 | `encoding=1 imputation=1 scaling=1 selection=3` |
| 40701 | `churn` | 0.8023 | 0 | 1.9 | `encoding=0 imputation=1 scaling=1 selection=1` |
| 1462 | `banknote-authentication` | 1.0000 | 0 | 0.9 | `encoding=2 imputation=1 scaling=3 selection=3` |
| 54 | `vehicle` | 0.7305 | 0 | 1.3 | `encoding=1 imputation=1 scaling=2 selection=3` |
| 46 | `splice` | 0.8126 | 8 | 6.0 | `encoding=0 imputation=2 scaling=3 selection=0` |
| 40994 | `climate-model-simulation-crashes` | 0.7353 | 0 | 1.2 | `encoding=1 imputation=1 scaling=0 selection=0` |
| 1489 | `phoneme` | 0.8307 | 0 | 1.1 | `encoding=2 imputation=1 scaling=2 selection=2` |
| 6332 | `cylinder-bands` | 0.8466 | 14 | 3.1 | `encoding=0 imputation=1 scaling=0 selection=3` |
| 31 | `credit-g` | 0.6754 | 8 | 1.8 | `encoding=0 imputation=0 scaling=2 selection=0` |
| 23381 | `dresses-sales` | 0.6458 | 8 | 2.0 | `encoding=0 imputation=1 scaling=1 selection=1` |
| 4538 | `GesturePhaseSegmentationProcessed` | 0.5799 | 0 | 2.5 | `encoding=2 imputation=1 scaling=3 selection=3` |
| 151 | `electricity` | 0.8364 | 0 | 11.4 | `encoding=1 imputation=1 scaling=2 selection=3` |
| 1494 | `qsar-biodeg` | 0.8582 | 0 | 2.1 | `encoding=0 imputation=1 scaling=0 selection=2` |
| 37 | `diabetes` | 0.6922 | 0 | 0.9 | `encoding=2 imputation=2 scaling=0 selection=2` |
