from PIL import Image
import numpy as np
import os

def check_mask_type(image_path):
    img = Image.open(image_path)
    print(f"File: {os.path.basename(image_path)}")
    print(f"Mode: {img.mode}")  # es: 'L' (grayscale), 'RGB', 'P', etc.
    print(f"Size: {img.size}")
    
    img_np = np.array(img)
    print(f"Numpy shape: {img_np.shape}")  # utile per capire se ha 1 o 3 canali

    # Se è RGB, stampa i valori unici per ogni canale
    if img_np.ndim == 3:
        print("Tipo: RGB")
        for i, channel in enumerate(['R', 'G', 'B']):
            unique_vals = np.unique(img_np[:, :, i])
            print(f"Valori unici nel canale {channel}: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
    else:
        print("Tipo: Grayscale")
        unique_vals = np.unique(img_np)
        print(f"Valori unici (classi?): {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
    
    print("-" * 40)

# 🔁 Per controllare più file in una cartella
folder = "./maschere"  # cambia questo con il tuo path
for file in os.listdir(folder):
    if file.endswith(".png"):
        check_mask_type(os.path.join(folder, file))
