# Corpus seleccionado (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\screen_corpus.py --seleccionar 25`.

- **Generado:** 2026-08-28
- **Procedimiento:** pre-registro v0.2, sección de corpus y cribado de sensibilidad
- **Candidatos cribados:** 52
- **Seleccionados:** 25
- **Semilla del cribado:** 4242

## Qué es este documento

La identidad del corpus es un **resultado** de aplicar el procedimiento
pre-registrado, no una regla, y por eso vive aquí y no en el pre-registro. Las
reglas que lo gobiernan quedaron fijadas en la v0.2, cerrada **antes** de esta
ejecución.

Cada candidato recibió doce pipelines extraídos al azar del espacio canónico con
la semilla 4242, descartando y volviendo a extraer los inevaluables hasta
reunir doce válidos. La medida de sensibilidad es el IQR de esos doce valores de
macro-F1. Los valores completos, con las configuraciones que los produjeron, están
en `CORPUS-cribado-valores.json`.

## Corpus

| # | OpenML ID | Nombre | IQR | Instancias | Categóricas | Clases | Faltantes | Descartadas | Licencia |
|---:|---:|---|---:|---:|---:|---:|:---:|---:|---|
| 1 | 40983 | `wilt` | 0.3239 | 4839 | 0 | 2 | no | 0 | Public |
| 2 | 38 | `sick` | 0.3064 | 3772 | 23 | 2 | sí | 14 | Public |
| 3 | 18 | `mfeat-morphological` | 0.2837 | 2000 | 0 | 10 | no | 0 | Public |
| 4 | 40975 | `car` | 0.2276 | 1728 | 6 | 4 | no | 7 | Public |
| 5 | 50 | `tic-tac-toe` | 0.2225 | 958 | 9 | 2 | no | 7 | Public |
| 6 | 40982 | `steel-plates-fault` | 0.1827 | 1941 | 0 | 7 | no | 0 | Public |
| 7 | 188 | `eucalyptus` | 0.1716 | 736 | 5 | 5 | sí | 14 | Public |
| 8 | 29 | `credit-approval` | 0.1490 | 690 | 9 | 2 | sí | 14 | Public |
| 9 | 11 | `balance-scale` | 0.1481 | 625 | 0 | 3 | no | 0 | Public |
| 10 | 1068 | `pc1` | 0.1475 | 1109 | 0 | 2 | no | 0 | Public |
| 11 | 41027 | `jungle_chess_2pcs_raw_endgame_complete` | 0.1430 | 44819 | 0 | 3 | no | 0 | Public |
| 12 | 1464 | `blood-transfusion-service-center` | 0.1180 | 748 | 0 | 2 | no | 0 | Public |
| 13 | 40701 | `churn` | 0.1019 | 5000 | 4 | 2 | no | 0 | public |
| 14 | 1462 | `banknote-authentication` | 0.1008 | 1372 | 0 | 2 | no | 0 | Public |
| 15 | 54 | `vehicle` | 0.0937 | 846 | 0 | 4 | no | 0 | Public |
| 16 | 46 | `splice` | 0.0897 | 3190 | 60 | 3 | no | 7 | Public |
| 17 | 40994 | `climate-model-simulation-crashes` | 0.0808 | 540 | 0 | 2 | no | 0 | Public |
| 18 | 1489 | `phoneme` | 0.0660 | 5404 | 0 | 2 | no | 0 | Public |
| 19 | 6332 | `cylinder-bands` | 0.0654 | 540 | 19 | 2 | sí | 14 | Public |
| 20 | 31 | `credit-g` | 0.0616 | 1000 | 13 | 2 | no | 7 | Public |
| 21 | 23381 | `dresses-sales` | 0.0589 | 500 | 11 | 2 | sí | 7 | Public |
| 22 | 4538 | `GesturePhaseSegmentationProcessed` | 0.0575 | 9873 | 0 | 5 | no | 0 | Public |
| 23 | 151 | `electricity` | 0.0563 | 45312 | 1 | 2 | no | 0 | Public |
| 24 | 1494 | `qsar-biodeg` | 0.0502 | 1055 | 0 | 2 | no | 0 | Public |
| 25 | 37 | `diabetes` | 0.0448 | 768 | 0 | 2 | no | 0 | Public |

**Subconjunto sensible al preprocesamiento** (IQR ≥ 0.02):
25 de 25.

## Restricciones de diversidad

| Restricción | Exigido | En el corpus | ¿Cumple? |
|---|---:|---:|:---:|
| Régimen de centenas | 1 | 10 | sí |
| Régimen de millares | 1 | 13 | sí |
| Régimen de decenas de millares | 1 | 2 | sí |
| Con variables categóricas | 5 | 11 | sí |
| Con valores faltantes | 5 | 5 | sí |

## Excluidos por la regla de familia

Como máximo un dataset por familia de origen, aplicada sobre el orden por IQR y
antes de las restricciones de diversidad.

- `mfeat-zernike` (IQR 0.0577): ya había un representante de la familia «dígitos manuscritos»
- `pendigits` (IQR 0.0566): ya había un representante de la familia «dígitos manuscritos»
- `mfeat-fourier` (IQR 0.0476): ya había un representante de la familia «dígitos manuscritos»
- `kc1` (IQR 0.0369): ya había un representante de la familia «NASA MDP»
- `pc3` (IQR 0.0311): ya había un representante de la familia «NASA MDP»
- `mfeat-karhunen` (IQR 0.0282): ya había un representante de la familia «dígitos manuscritos»
- `kc2` (IQR 0.0227): ya había un representante de la familia «NASA MDP»
- `optdigits` (IQR 0.0225): ya había un representante de la familia «dígitos manuscritos»
- `pc4` (IQR 0.0220): ya había un representante de la familia «NASA MDP»
- `jm1` (IQR 0.0180): ya había un representante de la familia «NASA MDP»

## Candidatos no seleccionados

La columna «motivo» distingue los excluidos por la regla de familia de los que
simplemente quedaron por debajo del corte en el orden por IQR.

| OpenML ID | Nombre | IQR | Motivo |
|---:|---|---:|---|
| 22 | `mfeat-zernike` | 0.0577 | familia |
| 32 | `pendigits` | 0.0566 | familia |
| 14 | `mfeat-fourier` | 0.0476 | familia |
| 23 | `cmc` | 0.0433 | orden |
| 1461 | `bank-marketing` | 0.0425 | orden |
| 1497 | `wall-robot-navigation` | 0.0384 | orden |
| 1067 | `kc1` | 0.0369 | familia |
| 1590 | `adult` | 0.0359 | orden |
| 1050 | `pc3` | 0.0311 | familia |
| 1480 | `ilpd` | 0.0310 | orden |
| 1510 | `wdbc` | 0.0301 | orden |
| 16 | `mfeat-karhunen` | 0.0282 | familia |
| 182 | `satimage` | 0.0271 | orden |
| 40966 | `MiceProtein` | 0.0241 | orden |
| 1063 | `kc2` | 0.0227 | familia |
| 28 | `optdigits` | 0.0225 | familia |
| 1049 | `pc4` | 0.0220 | familia |
| 469 | `analcatdata_dmft` | 0.0207 | orden |
| 1053 | `jm1` | 0.0180 | familia |
| 1475 | `first-order-theorem-proving` | 0.0173 | orden |
| 458 | `analcatdata_authorship` | 0.0154 | orden |
| 44 | `spambase` | 0.0152 | orden |
| 1487 | `ozone-level-8hr` | 0.0143 | orden |
| 3 | `kr-vs-kp` | 0.0110 | orden |
| 40984 | `segment` | 0.0072 | orden |
| 15 | `breast-w` | 0.0049 | orden |
| 4534 | `PhishingWebsites` | 0.0027 | orden |
