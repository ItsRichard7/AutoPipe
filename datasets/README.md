# datasets/

Datos locales usados por los notebooks de `examples/` y por los scripts de
verificación. **Su contenido no se versiona**: los datasets son grandes, y sus licencias
condicionan o prohíben redistribuirlos.

## Dónde ponerlos: fuera de OneDrive

La carpeta del proyecto vive dentro de OneDrive, así que **cualquier cosa que se
guarde aquí se sincroniza**. Para los datasets eso es un problema, por tres razones
concretas:

1. **Bloqueos de archivo.** Ya ocurrió una vez: un `git pull` falló porque OneDrive
   tenía tomado un archivo (bitácora del 2026-08-03). Con miles de imágenes, la
   probabilidad de que vuelva a pasar en mitad de un experimento sube mucho.
2. **Archivos a petición.** OneDrive puede dejar los archivos como marcadores de
   posición y bajarlos al abrirlos. Si uno está deshidratado y la sincronización
   está lenta o caída, la lectura falla y el experimento se cae por una razón que
   no tiene nada que ver con el código.
3. **Cuota.** Sincronizar imágenes que nunca cambian gasta cuota institucional sin
   ningún beneficio: el dataset se recupera volviéndolo a descargar.

**Recomendado:** guardarlos en una ruta local fuera del árbol sincronizado, por
ejemplo `C:\datasets\`, y apuntar la variable de entorno `AUTOPIPE_DATASETS` a esa
carpeta:

```powershell
$env:AUTOPIPE_DATASETS = "C:\datasets"
```

Si la variable no está definida, los scripts usan esta carpeta del repositorio como
respaldo.

## Estructura esperada

```
<AUTOPIPE_DATASETS>/
├── mvtec_ad/              # MVTec AD
│   └── bottle/
│       ├── test/          # <- lo único que usan los scripts
│       │   ├── good/              (normales)
│       │   ├── broken_large/      (anómalas)
│       │   ├── broken_small/
│       │   └── contamination/
│       ├── train/         # no se usa; se puede borrar
│       └── ground_truth/  # no se usa; se puede borrar
└── cifar-10-batches-py/   # CIFAR-10 (torchvision lo descarga aquí)
```

`ImageDataLoader` espera `normal/` y `anomaly/`, que no es la estructura nativa de
MVTec AD. La conversión la hacen el notebook `examples/mvtec_anomaly_detection.ipynb`
y `scripts/check_image_regression.py`: el split `test/` ya mezcla normales
(`good/`) y anómalas (una carpeta por tipo de defecto), que es justo lo que hace
falta.

## MVTec AD

- Descarga: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- Requiere registro y aceptar la licencia **CC BY-NC-SA 4.0** (uso no comercial de
  investigación). Un TFG entra sin problema. La licencia **no prohíbe redistribuir** las
  imágenes: las condiciona a un uso no comercial, con atribución y bajo la misma
  licencia. Aun así, **este proyecto opta por no redistribuirlas** —son datos de un
  tercero, pesan, y quien los quiera los descarga del origen—, y por eso esta carpeta
  está en `.gitignore`.
- Hay descarga por categoría además del paquete completo. Basta con `bottle`.
- **Citar** en el informe: Bergmann et al. (2019), *MVTec AD — A Comprehensive
  Real-World Dataset for Unsupervised Anomaly Detection*, CVPR.

Extraer con (GNU tar, incluido en Git Bash):

```bash
tar -xf bottle.tar.xz -C /c/datasets/mvtec_ad
```

La categoría `bottle` trae 83 imágenes en `test/`: 20 normales y 63 anómalas
repartidas en tres tipos de defecto, todas de 900x900 en color.

## Nota

Los notebooks funcionan **sin** estos datos: si no encuentran el dataset real,
generan un sustituto sintético para ejecutarse de extremo a extremo. Los scripts de
verificación, en cambio, exigen los datos reales y fallan con un mensaje explícito
si no están.

Las pruebas unitarias **no** dependen de esta carpeta: sus imágenes se generan al
vuelo (véase `tests/conftest.py`).
