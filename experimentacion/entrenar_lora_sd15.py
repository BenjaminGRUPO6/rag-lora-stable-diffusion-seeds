import os
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class SoybeanDataset(Dataset):
    """
    Dataset simple para cargar imágenes y su prompt.
    Asume una carpeta con imágenes (.jpg, .png) donde todas usarán el mismo prompt,
    o prompts basados en el nombre de la carpeta/archivo.
    Para este ejemplo, usaremos un prompt fijo o por clase.
    """
    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.rglob("*.jpg")) + list(self.data_dir.rglob("*.png"))
        self.tokenizer = tokenizer
        self.size = size
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        

        clase = img_path.parent.name
        prompt = f"a photo of a {clase} soybeanseed"
        
        # Tokenizar el prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image_tensor,
            "input_ids": text_inputs.input_ids[0]
        }


def main():
    parser = argparse.ArgumentParser(description="Entrenar LoRA para SD 1.5")
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta a la carpeta con tus imágenes reales")
    parser.add_argument("--output_dir", type=str, default="lora_output", help="Dónde guardar el modelo")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas (vueltas al dataset)")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamaño de batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Tasa de aprendizaje")
    args = parser.parse_args()

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    # 1. Iniciar Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    print(f"Entrenando usando: {device}")

    # 2. Cargar componentes de Stable Diffusion
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Congelar pesos base
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. Configurar LoRA en la UNet (el cerebro de la imagen)
    lora_config = LoraConfig(
        r=8, # Rango del LoRA (tamaño)
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"] # Inyectar en capas de atención
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 4. Optimizador
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # 5. Dataset y Dataloader
    dataset = SoybeanDataset(args.data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 6. Preparar con Accelerate
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    # Mover componentes congelados a la GPU
    text_encoder.to(device)
    vae.to(device)

    # 7. Bucle de Entrenamiento
    print("¡Comenzando entrenamiento!")
    global_step = 0
    
    for epoch in range(args.epochs):
        unet.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            # Obtener latentes de la imagen
            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Crear ruido aleatorio
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            # Añadir ruido a la imagen (Forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Obtener embeddings del texto (el prompt)
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

            # Predecir el ruido residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calcular la pérdida (MSE entre ruido predicho y ruido real)
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            epoch_loss += loss.item()

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"Época {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

    # 8. Guardar el LoRA resultante
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_unet.save_pretrained(args.output_dir)
        print(f"Entrenamiento culminad, modelo LoRA guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
