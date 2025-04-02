from PIL import Image
import numpy as np

def validate_grayscale_mask(filename, n_min, n_max):
    try:
        img = Image.open(filename)
        print(f"Verifica file: {filename}")
        
        # Controlla che sia in scala di grigi (modalità 'L' = 8-bit grayscale)
        if img.mode != 'L':
            print("❌ L'immagine NON è in scala di grigi. Mode:", img.mode)
            return False
        
        # Converte in array NumPy e controlla i valori
        img_np = np.array(img)
        
        if not np.issubdtype(img_np.dtype, np.integer):
            print("❌ I valori nell'immagine non sono interi.")
            return False

        unique_vals = np.unique(img_np)
        if unique_vals.min() < n_min or unique_vals.max() > n_max:
            print(f"❌ Valori fuori dal range [{n_min}, {n_max}].")
            print(f"Valori unici trovati: {unique_vals}")
            return False
        
        print("✅ Immagine valida: scala di grigi e valori nel range.")
        return True

    except Exception as e:
        print("⚠️ Errore durante il controllo:", e)
        return False


filepath = '/outputs/airflow_data/floodnet/masks/7684_lab.png'

validate_grayscale_mask(filepath, 0, 9)