"""
1_generar_sinteticas.py
=======================
Genera imágenes sintéticas de semillas de soja usando Stable Diffusion 1.5 + LoRA.

Mejoras respecto a la versión anterior:
- Filtro NSFW desactivado en src/synthetic_data/generate_images.py (falsos positivos agrícolas).
- Prompts positivos más específicos y descriptivos por clase.
- Negative prompt global para evitar artefactos comunes.
- 5 imágenes por clase con semillas diversas.
- Reintentos automáticos con semilla alternativa si la imagen resulta completamente negra.

Uso:
    .venv\\Scripts\\python.exe experimentacion/1_generar_sinteticas.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Añadir la raíz del proyecto al sys.path para importar src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.synthetic_data.generate_images import generate, load_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGES_PER_CLASS = 5
OUTPUT_DIR = project_root / "experimentacion" / "sinteticas_crudas"
MAX_RETRIES = 3  # intentos extra con semilla distinta si la imagen sale negra

# Negative prompt compartido: evita artefactos, colores incorrectos y contenido
# ajeno a fotografías de semillas individuales sobre fondo neutro.
NEGATIVE_PROMPT = (
    "blurry, out of focus, dark image, black image, oversaturated, "
    "multiple seeds, hands, text, watermark, cartoon, illustration, "
    "painting, render, 3d, noise, grain, low quality, deformed"
)

# Prompts positivos por clase — incluyen el trigger word "soybeanseed" y
# descripciones visuales específicas que guían al LoRA.
PROMPTS: dict[str, str] = {
    "intact": (
        "close-up photo of a single intact soybeanseed, perfectly round, "
        "smooth yellow surface, healthy soybean, studio lighting, white background, "
        "macro photography, sharp focus, highly detailed"
    ),
    "spotted": (
        "close-up photo of a single spotted soybeanseed, dark brown spots "
        "scattered on the surface, discolored patches, yellow soybean with markings, "
        "white background, macro photography, sharp focus, highly detailed"
    ),
    "immature": (
        "close-up photo of a single immature soybeanseed, small greenish bean, "
        "slightly wrinkled, pale green color, underdeveloped seed, "
        "white background, macro photography, sharp focus, highly detailed"
    ),
    "broken": (
        "close-up photo of a single broken soybeanseed, cracked seed, split in half, "
        "fractured surface, rough edges, yellow interior exposed, "
        "white background, macro photography, sharp focus, highly detailed"
    ),
    "skin_damaged": (
        "close-up photo of a single skin_damaged soybeanseed, peeled skin, "
        "wrinkled outer layer, damaged hull, rough texture, abraded surface, "
        "white background, macro photography, sharp focus, highly detailed"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_black_image(image: Image.Image, threshold: float = 0.02) -> bool:
    """Devuelve True si la imagen es mayormente negra (salida del filtro NSFW)."""
    arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return float(arr.mean()) < threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    lora_path = project_root / "models" / "lora" / "soybean_sd15" / "pytorch_lora_weights.safetensors"

    if not lora_path.exists():
        print(f"Error: No se encontró el adaptador LoRA en {lora_path}")
        return

    print("Cargando modelo y LoRA...")
    pipe = load_pipeline(BASE_MODEL, lora_path)
    print("Modelo cargado exitosamente.\n")

    total_generadas = 0
    total_negras = 0

    for category, prompt in PROMPTS.items():
        cat_dir = OUTPUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"  Categoría: {category.upper()}")
        print(f"{'='*50}")

        generadas_clase = 0
        imagen_idx = 0

        for i in range(IMAGES_PER_CLASS):
            output_path = cat_dir / f"synth_{i:03d}.jpg"
            base_seed = 100 + i * 37 + abs(hash(category)) % 5000

            imagen_ok = False
            for intento in range(MAX_RETRIES):
                seed = base_seed + intento * 1000
                print(f"  [{i+1:>2}/{IMAGES_PER_CLASS}] {output_path.name}  seed={seed}", end=" ... ")

                try:
                    generate(
                        pipe,
                        prompt=prompt,
                        output_path=output_path,
                        negative_prompt=NEGATIVE_PROMPT,
                        steps=35,
                        guidance_scale=8.0,
                        seed=seed,
                    )
                    # Verificar que la imagen no sea negra
                    with Image.open(output_path) as img:
                        if is_black_image(img):
                            print(f"NEGRA (intento {intento+1}/{MAX_RETRIES})", end=" ")
                            total_negras += 1
                            if intento < MAX_RETRIES - 1:
                                print("reintentando...")
                                continue
                            else:
                                print("sin éxito — guardada de todas formas.")
                        else:
                            print("OK")
                            imagen_ok = True
                            generadas_clase += 1
                            break

                except Exception as exc:
                    print(f"ERROR: {exc}")
                    break

            if imagen_ok:
                total_generadas += 1
            imagen_idx += 1

        print(f"  -> {generadas_clase}/{IMAGES_PER_CLASS} imágenes válidas para '{category}'")

    print(f"\n{'='*50}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"  Imágenes válidas  : {total_generadas}/{IMAGES_PER_CLASS * len(PROMPTS)}")
    print(f"  Imágenes negras   : {total_negras} (falsos positivos del filtro NSFW desactivado)")
    print(f"  Directorio salida : {OUTPUT_DIR}")
    print(f"\nRevisa las imágenes y copia las útiles a:")
    print(f"  experimentacion/sinteticas_aprobadas/<clase>/")


if __name__ == "__main__":
    main()
