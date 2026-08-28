# Corpus descargado y verificado (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\download_corpus.py`.

- **Generado:** 2026-08-28
- **Entrada:** `docs/experimentos/CORPUS-candidatos.md`
- **Datasets descargados:** 52
- **Con discrepancias en los metadatos:** 0

## Por qué existe este documento

La lista de candidatos se construyó con los metadatos que publica OpenML, sin
descargar ningún dataset. Este paso los comprueba contra los datos reales, que es
la limitación que el documento de criterios dejó anotada para la semana 5.

## Diferencias explicadas por atributos ignorados

`NumberOfFeatures` cuenta también los atributos que OpenML marca como
identificadores o ignorables y que descarta al entregar la tabla. La diferencia es
esperada, y descartarlos es lo correcto: una columna identificadora no aporta
señal y puede introducir una correlación espuria.

- `cylinder-bands` (ID 6332): OpenML ignora `timestamp`, `cylinder_number`
- `climate-model-simulation-crashes` (ID 40994): OpenML ignora `Study`, `Run`
- `MiceProtein` (ID 40966): OpenML ignora `Genotype`, `Treatment`, `Behavior`, `MouseID`
- `segment` (ID 40984): OpenML ignora `region.centroid.col`, `region.centroid.row`, `region.pixel.count`

## Columnas enteramente vacías

Una columna sin ningún valor llega como `object` y se cuenta como categórica,
aunque OpenML la declare numérica. **No se descartan**: el pipeline las trata
correctamente —la imputación las vuelve constantes y la selección supervisada las
puntúa como las peores— y eliminarlas introduciría una transformación silenciosa
entre lo que dice el pre-registro y lo que se ejecuta.

- `sick` (ID 38): `TBG`

## Discrepancias sin explicar

Ninguna: los metadatos de OpenML coinciden con los datos en los 52 datasets.

## Meta-características no finitas

Cada dataset aporta las columnas que le resultan no finitas. La regla del
pre-registro descarta las que lo sean en **algún** dataset del corpus, de modo que
el vector final es la intersección de las finitas.

- Meta-características extraídas por dataset: **107**
- No finitas en al menos un dataset: **27**
- **Sobrevivirían: 80**

## Cobertura de diversidad, medida sobre los datos

El pre-registro exige, al elegir el corpus final, cubrir los tres regímenes de
tamaño y contar con al menos cinco datasets con variables categóricas y cinco con
valores faltantes. Estas cuentas ya no salen de los metadatos de OpenML sino de
los datos descargados.

| Restricción | Exigido | Disponible |
|---|---:|---:|
| Régimen de centenas | 1 | 16 |
| Régimen de millares | 1 | 29 |
| Régimen de decenas de millares | 1 | 7 |
| Con variables categóricas | 5 | 18 |
| Con valores faltantes | 5 | 9 |
| Multiclase (más de 2 clases) | 0 | 22 |

## Fichas verificadas

Los valores son los **reales**, medidos sobre los datos descargados. La columna
«Metadatos» dice si coinciden con lo que OpenML declaraba.

| OpenML ID | Nombre | Instancias | Predictoras | Categóricas | Clases | Faltantes | No finitas | Licencia | Metadatos |
|---:|---|---:|---:|---:|---:|:---:|---:|---|---|
| 23381 | `dresses-sales` | 500 | 12 | 11 | 2 | sí | 10 | Public | OK |
| 1063 | `kc2` | 522 | 21 | 0 | 2 | no | 12 | Public | OK |
| 6332 | `cylinder-bands` | 540 | 37 | 19 | 2 | sí | 10 | Public | ignora 2 |
| 40994 | `climate-model-simulation-crashes` | 540 | 18 | 0 | 2 | no | 11 | Public | ignora 2 |
| 1510 | `wdbc` | 569 | 30 | 0 | 2 | no | 11 | Public | OK |
| 1480 | `ilpd` | 583 | 10 | 1 | 2 | no | 10 | Public | OK |
| 11 | `balance-scale` | 625 | 4 | 0 | 3 | no | 3 | Public | OK |
| 29 | `credit-approval` | 690 | 15 | 9 | 2 | sí | 10 | Public | OK |
| 15 | `breast-w` | 699 | 9 | 0 | 2 | sí | 10 | Public | OK |
| 188 | `eucalyptus` | 736 | 19 | 5 | 5 | sí | 9 | Public | OK |
| 1464 | `blood-transfusion-service-center` | 748 | 4 | 0 | 2 | no | 13 | Public | OK |
| 37 | `diabetes` | 768 | 8 | 0 | 2 | no | 10 | Public | OK |
| 469 | `analcatdata_dmft` | 797 | 4 | 4 | 6 | no | 6 | Public | OK |
| 458 | `analcatdata_authorship` | 841 | 70 | 0 | 4 | no | 4 | Public | OK |
| 54 | `vehicle` | 846 | 18 | 0 | 4 | no | 4 | Public | OK |
| 50 | `tic-tac-toe` | 958 | 9 | 9 | 2 | no | 11 | Public | OK |
| 31 | `credit-g` | 1000 | 20 | 13 | 2 | no | 10 | Public | OK |
| 1494 | `qsar-biodeg` | 1055 | 41 | 0 | 2 | no | 15 | Public | OK |
| 40966 | `MiceProtein` | 1080 | 77 | 0 | 8 | sí | 8 | Public | ignora 4 |
| 1068 | `pc1` | 1109 | 21 | 0 | 2 | no | 11 | Public | OK |
| 1462 | `banknote-authentication` | 1372 | 4 | 0 | 2 | no | 14 | Public | OK |
| 1049 | `pc4` | 1458 | 37 | 0 | 2 | no | 11 | Public | OK |
| 23 | `cmc` | 1473 | 9 | 7 | 3 | no | 5 | Public | OK |
| 1050 | `pc3` | 1563 | 37 | 0 | 2 | no | 11 | Public | OK |
| 40975 | `car` | 1728 | 6 | 6 | 4 | no | 6 | Public | OK |
| 40982 | `steel-plates-fault` | 1941 | 27 | 0 | 7 | no | 10 | Public | OK |
| 14 | `mfeat-fourier` | 2000 | 76 | 0 | 10 | no | 3 | Public | OK |
| 18 | `mfeat-morphological` | 2000 | 6 | 0 | 10 | no | 6 | Public | OK |
| 22 | `mfeat-zernike` | 2000 | 47 | 0 | 10 | no | 4 | Public | OK |
| 16 | `mfeat-karhunen` | 2000 | 64 | 0 | 10 | no | 7 | Public | OK |
| 1067 | `kc1` | 2109 | 21 | 0 | 2 | no | 12 | Public | OK |
| 40984 | `segment` | 2310 | 16 | 0 | 7 | no | 11 | Public | ignora 3 |
| 1487 | `ozone-level-8hr` | 2534 | 72 | 0 | 2 | no | 14 | Public | OK |
| 46 | `splice` | 3190 | 60 | 60 | 3 | no | 6 | Public | OK |
| 3 | `kr-vs-kp` | 3196 | 36 | 36 | 2 | no | 11 | Public | OK |
| 38 | `sick` | 3772 | 29 | 23 | 2 | sí | 10 | Public | OK |
| 44 | `spambase` | 4601 | 57 | 0 | 2 | no | 10 | Public | OK |
| 40983 | `wilt` | 4839 | 5 | 0 | 2 | no | 10 | Public | OK |
| 40701 | `churn` | 5000 | 20 | 4 | 2 | no | 10 | public | OK |
| 1489 | `phoneme` | 5404 | 5 | 0 | 2 | no | 14 | Public | OK |
| 1497 | `wall-robot-navigation` | 5456 | 24 | 0 | 4 | no | 3 | Public | OK |
| 28 | `optdigits` | 5620 | 64 | 0 | 10 | no | 14 | Public | OK |
| 1475 | `first-order-theorem-proving` | 6118 | 51 | 0 | 6 | no | 10 | Public | OK |
| 182 | `satimage` | 6430 | 36 | 0 | 6 | no | 7 | Public | OK |
| 4538 | `GesturePhaseSegmentationProcessed` | 9873 | 32 | 0 | 5 | no | 7 | Public | OK |
| 1053 | `jm1` | 10885 | 21 | 0 | 2 | sí | 10 | Public | OK |
| 32 | `pendigits` | 10992 | 16 | 0 | 10 | no | 4 | Public | OK |
| 4534 | `PhishingWebsites` | 11055 | 30 | 30 | 2 | no | 13 | Public | OK |
| 41027 | `jungle_chess_2pcs_raw_endgame_complete` | 44819 | 6 | 0 | 3 | no | 4 | Public | OK |
| 1461 | `bank-marketing` | 45211 | 16 | 9 | 2 | no | 14 | Public | OK |
| 151 | `electricity` | 45312 | 8 | 1 | 2 | no | 10 | Public | OK |
| 1590 | `adult` | 48842 | 14 | 8 | 2 | sí | 10 | Public | OK |
