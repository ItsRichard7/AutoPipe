# datasets/

Carpeta local (gitignored) para los datasets usados por los notebooks de `examples/`.
Su contenido **no se versiona**: los datasets son grandes y/o requieren descarga
propia.

Estructura esperada:

```
datasets/
├── mvtec_ad/          # MVTec AD (descarga: https://www.mvtec.com/company/research/datasets/mvtec-ad)
│   ├── bottle/
│   ├── cable/
│   └── ...
└── cifar-10-batches-py/   # CIFAR-10 (torchvision lo descarga aquí automáticamente)
```

Los notebooks funcionan sin estos datos: si no encuentran el dataset real, generan
un sustituto sintético para ejecutarse de extremo a extremo.

Nota: los tests unitarios **no** dependen de esta carpeta; sus imágenes de prueba
se generan al vuelo (ver `tests/conftest.py`).
