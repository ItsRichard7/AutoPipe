# Corpus del experimento 2 (42 conjuntos)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\build_corpus_e2.py`.

- **Generado:** 2026-09-08
- **Entrada:** `docs/experimentos/CORPUS-seleccionado.md`
- **Gobierna:** pre-registro del experimento 2, v2.0, etiqueta `prereg-v2.0`
- **Composición:** 25 del experimento 1 + 17
  incorporados = **42**

## Por qué 42 y no 52

El cribado descargó y verificó 52 candidatos y seleccionó 25. De los 27 restantes,
**10 quedaron fuera por la regla de familia y siguen fuera**; entran
los 17 que solo quedaron por debajo del corte en el orden por
sensibilidad.

En una validación por exclusión de conjunto, admitir dos conjuntos de la misma
familia deja al casi gemelo entrenando al recomendador justo en el pliegue que
excluye a su pareja. El vecino más cercano sería un hermano, y el arranque en
caliente parecería acertar por una similitud que no es la que el método pretende
explotar. **El sesgo iría a favor de la hipótesis**, que es la única dirección en la
que no se puede admitir ninguno.

Los criterios de inclusión del anteproyecto se mantienen sin alteración: los 42 ya
los cumplen. Lo único que se retira es el corte por sensibilidad, que existía para
quedarse con 25.

## Corpus

Ordenado por sensibilidad al preprocesamiento descendente, que es el orden que
produjo el cribado.

| # | OpenML ID | Nombre | IQR | Procedencia |
|---:|---:|---|---:|---|
| 1 | 40983 | `wilt` | 0.3239 | corpus del experimento 1 |
| 2 | 38 | `sick` | 0.3064 | corpus del experimento 1 |
| 3 | 18 | `mfeat-morphological` | 0.2837 | corpus del experimento 1 |
| 4 | 40975 | `car` | 0.2276 | corpus del experimento 1 |
| 5 | 50 | `tic-tac-toe` | 0.2225 | corpus del experimento 1 |
| 6 | 40982 | `steel-plates-fault` | 0.1827 | corpus del experimento 1 |
| 7 | 188 | `eucalyptus` | 0.1716 | corpus del experimento 1 |
| 8 | 29 | `credit-approval` | 0.1490 | corpus del experimento 1 |
| 9 | 11 | `balance-scale` | 0.1481 | corpus del experimento 1 |
| 10 | 1068 | `pc1` | 0.1475 | corpus del experimento 1 |
| 11 | 41027 | `jungle_chess_2pcs_raw_endgame_complete` | 0.1430 | corpus del experimento 1 |
| 12 | 1464 | `blood-transfusion-service-center` | 0.1180 | corpus del experimento 1 |
| 13 | 40701 | `churn` | 0.1019 | corpus del experimento 1 |
| 14 | 1462 | `banknote-authentication` | 0.1008 | corpus del experimento 1 |
| 15 | 54 | `vehicle` | 0.0937 | corpus del experimento 1 |
| 16 | 46 | `splice` | 0.0897 | corpus del experimento 1 |
| 17 | 40994 | `climate-model-simulation-crashes` | 0.0808 | corpus del experimento 1 |
| 18 | 1489 | `phoneme` | 0.0660 | corpus del experimento 1 |
| 19 | 6332 | `cylinder-bands` | 0.0654 | corpus del experimento 1 |
| 20 | 31 | `credit-g` | 0.0616 | corpus del experimento 1 |
| 21 | 23381 | `dresses-sales` | 0.0589 | corpus del experimento 1 |
| 22 | 4538 | `GesturePhaseSegmentationProcessed` | 0.0575 | corpus del experimento 1 |
| 23 | 151 | `electricity` | 0.0563 | corpus del experimento 1 |
| 24 | 1494 | `qsar-biodeg` | 0.0502 | corpus del experimento 1 |
| 25 | 37 | `diabetes` | 0.0448 | corpus del experimento 1 |
| 26 | 23 | `cmc` | 0.0433 | incorporado en el experimento 2 |
| 27 | 1461 | `bank-marketing` | 0.0425 | incorporado en el experimento 2 |
| 28 | 1497 | `wall-robot-navigation` | 0.0384 | incorporado en el experimento 2 |
| 29 | 1590 | `adult` | 0.0359 | incorporado en el experimento 2 |
| 30 | 1480 | `ilpd` | 0.0310 | incorporado en el experimento 2 |
| 31 | 1510 | `wdbc` | 0.0301 | incorporado en el experimento 2 |
| 32 | 182 | `satimage` | 0.0271 | incorporado en el experimento 2 |
| 33 | 40966 | `MiceProtein` | 0.0241 | incorporado en el experimento 2 |
| 34 | 469 | `analcatdata_dmft` | 0.0207 | incorporado en el experimento 2 |
| 35 | 1475 | `first-order-theorem-proving` | 0.0173 | incorporado en el experimento 2 |
| 36 | 458 | `analcatdata_authorship` | 0.0154 | incorporado en el experimento 2 |
| 37 | 44 | `spambase` | 0.0152 | incorporado en el experimento 2 |
| 38 | 1487 | `ozone-level-8hr` | 0.0143 | incorporado en el experimento 2 |
| 39 | 3 | `kr-vs-kp` | 0.0110 | incorporado en el experimento 2 |
| 40 | 40984 | `segment` | 0.0072 | incorporado en el experimento 2 |
| 41 | 15 | `breast-w` | 0.0049 | incorporado en el experimento 2 |
| 42 | 4534 | `PhishingWebsites` | 0.0027 | incorporado en el experimento 2 |

## Siguen excluidos por la regla de familia

| OpenML ID | Nombre | IQR |
|---:|---|---:|
| 22 | `mfeat-zernike` | 0.0577 |
| 32 | `pendigits` | 0.0566 |
| 14 | `mfeat-fourier` | 0.0476 |
| 1067 | `kc1` | 0.0369 |
| 1050 | `pc3` | 0.0311 |
| 16 | `mfeat-karhunen` | 0.0282 |
| 1063 | `kc2` | 0.0227 |
| 28 | `optdigits` | 0.0225 |
| 1049 | `pc4` | 0.0220 |
| 1053 | `jm1` | 0.0180 |
