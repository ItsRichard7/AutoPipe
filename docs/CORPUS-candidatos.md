# Lista candidata de datasets para el corpus (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\list_corpus_candidates.py`.

- **Generado:** 2026-08-24
- **Universo:** suite curada OpenML-CC18 (72 datasets)
- **Criterios aplicados:** Sección 4.3.1 del anteproyecto
- **Decisiones de procedimiento:** `docs/CORPUS-criterios-inclusion.md`

## Resumen

- Candidatos que superan los criterios: **52**
- Por régimen de tamaño:
  - centenas: 16
  - millares: 29
  - decenas de millares: 7
- Con variables categóricas: 19
- Con valores faltantes: 9
- Multiclase (más de 2 clases): 22

## Fichas

«Predictoras» y «categóricas» excluyen la columna objetivo. «Desbalance» es
el cociente entre la clase mayoritaria y la minoritaria.

| OpenML ID | Nombre | Instancias | Predictoras | Categóricas | Clases | Faltantes | Desbalance | Régimen |
|---:|---|---:|---:|---:|---:|:---:|---:|---|
| 23381 | `dresses-sales` | 500 | 12 | 11 | 2 | sí | 1.4 | centenas |
| 1063 | `kc2` | 522 | 21 | 0 | 2 | no | 3.9 | centenas |
| 6332 | `cylinder-bands` | 540 | 39 | 21 | 2 | sí | 1.4 | centenas |
| 40994 | `climate-model-simulation-crashes` | 540 | 20 | 0 | 2 | no | 10.7 | centenas |
| 1510 | `wdbc` | 569 | 30 | 0 | 2 | no | 1.7 | centenas |
| 1480 | `ilpd` | 583 | 10 | 1 | 2 | no | 2.5 | centenas |
| 11 | `balance-scale` | 625 | 4 | 0 | 3 | no | 5.9 | centenas |
| 29 | `credit-approval` | 690 | 15 | 9 | 2 | sí | 1.2 | centenas |
| 15 | `breast-w` | 699 | 9 | 0 | 2 | sí | 1.9 | centenas |
| 188 | `eucalyptus` | 736 | 19 | 5 | 5 | sí | 2.0 | centenas |
| 1464 | `blood-transfusion-service-center` | 748 | 4 | 0 | 2 | no | 3.2 | centenas |
| 37 | `diabetes` | 768 | 8 | 0 | 2 | no | 1.9 | centenas |
| 469 | `analcatdata_dmft` | 797 | 4 | 4 | 6 | no | 1.3 | centenas |
| 458 | `analcatdata_authorship` | 841 | 70 | 0 | 4 | no | 5.8 | centenas |
| 54 | `vehicle` | 846 | 18 | 0 | 4 | no | 1.1 | centenas |
| 50 | `tic-tac-toe` | 958 | 9 | 9 | 2 | no | 1.9 | centenas |
| 31 | `credit-g` | 1000 | 20 | 13 | 2 | no | 2.3 | millares |
| 1494 | `qsar-biodeg` | 1055 | 41 | 0 | 2 | no | 2.0 | millares |
| 40966 | `MiceProtein` | 1080 | 81 | 4 | 8 | sí | 1.4 | millares |
| 1068 | `pc1` | 1109 | 21 | 0 | 2 | no | 13.4 | millares |
| 1462 | `banknote-authentication` | 1372 | 4 | 0 | 2 | no | 1.2 | millares |
| 1049 | `pc4` | 1458 | 37 | 0 | 2 | no | 7.2 | millares |
| 23 | `cmc` | 1473 | 9 | 7 | 3 | no | 1.9 | millares |
| 1050 | `pc3` | 1563 | 37 | 0 | 2 | no | 8.8 | millares |
| 40975 | `car` | 1728 | 6 | 6 | 4 | no | 18.6 | millares |
| 40982 | `steel-plates-fault` | 1941 | 27 | 0 | 7 | no | 12.2 | millares |
| 14 | `mfeat-fourier` | 2000 | 76 | 0 | 10 | no | 1.0 | millares |
| 18 | `mfeat-morphological` | 2000 | 6 | 0 | 10 | no | 1.0 | millares |
| 22 | `mfeat-zernike` | 2000 | 47 | 0 | 10 | no | 1.0 | millares |
| 16 | `mfeat-karhunen` | 2000 | 64 | 0 | 10 | no | 1.0 | millares |
| 1067 | `kc1` | 2109 | 21 | 0 | 2 | no | 5.5 | millares |
| 40984 | `segment` | 2310 | 19 | 0 | 7 | no | 1.0 | millares |
| 1487 | `ozone-level-8hr` | 2534 | 72 | 0 | 2 | no | 14.8 | millares |
| 46 | `splice` | 3190 | 60 | 60 | 3 | no | 2.2 | millares |
| 3 | `kr-vs-kp` | 3196 | 36 | 36 | 2 | no | 1.1 | millares |
| 38 | `sick` | 3772 | 29 | 22 | 2 | sí | 15.3 | millares |
| 44 | `spambase` | 4601 | 57 | 0 | 2 | no | 1.5 | millares |
| 40983 | `wilt` | 4839 | 5 | 0 | 2 | no | 17.5 | millares |
| 40701 | `churn` | 5000 | 20 | 4 | 2 | no | 6.1 | millares |
| 1489 | `phoneme` | 5404 | 5 | 0 | 2 | no | 2.4 | millares |
| 1497 | `wall-robot-navigation` | 5456 | 24 | 0 | 4 | no | 6.7 | millares |
| 28 | `optdigits` | 5620 | 64 | 0 | 10 | no | 1.0 | millares |
| 1475 | `first-order-theorem-proving` | 6118 | 51 | 0 | 6 | no | 5.3 | millares |
| 182 | `satimage` | 6430 | 36 | 0 | 6 | no | 2.4 | millares |
| 4538 | `GesturePhaseSegmentationProcessed` | 9873 | 32 | 0 | 5 | no | 3.0 | millares |
| 1053 | `jm1` | 10885 | 21 | 0 | 2 | sí | 4.2 | decenas de millares |
| 32 | `pendigits` | 10992 | 16 | 0 | 10 | no | 1.1 | decenas de millares |
| 4534 | `PhishingWebsites` | 11055 | 30 | 30 | 2 | no | 1.3 | decenas de millares |
| 41027 | `jungle_chess_2pcs_raw_endgame_complete` | 44819 | 6 | 0 | 3 | no | 5.3 | decenas de millares |
| 1461 | `bank-marketing` | 45211 | 16 | 9 | 2 | no | 7.5 | decenas de millares |
| 151 | `electricity` | 45312 | 8 | 1 | 2 | no | 1.4 | decenas de millares |
| 1590 | `adult` | 48842 | 14 | 8 | 2 | sí | 3.2 | decenas de millares |
