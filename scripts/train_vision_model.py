"""Entrada del entrenamiento visual.

Complete el notebook 04 o implemente la conexión de DataLoaders aquí.
El código reutilizable se encuentra en src/vision/.
"""

from src.utils.config import load_yaml
from src.vision.model import create_model


def main() -> None:
    config = load_yaml("configs/vision_config.yaml")
    model = create_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )
    print(model.__class__.__name__)
    print("Modelo creado. Continúe con DataLoaders y el bucle de entrenamiento en el notebook 04.")


if __name__ == "__main__":
    main()
