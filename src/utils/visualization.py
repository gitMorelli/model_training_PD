import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def debug_images_dataset(dataset, output_path="anteprima_dataset.png", num_immagini=16, mean=None, std=None):
    """
    Estrae un numero specifico di immagini da un WebDataset (o standard Dataset) 
    iterando sugli elementi e le salva in una griglia su file.
    """
    # 1. Creiamo un DataLoader temporaneo (batch_size=None mantiene lo streaming nativo)
    dataloader = DataLoader(
        dataset, 
        num_workers=0, 
        batch_size=None, 
        prefetch_factor=None,
    )
    
    immagini_raccolte = []
    data_iter = iter(dataloader)
    
    # 2. Iteriamo ed estraiamo campioni finché non raggiungiamo 'num_immagini'
    print(f"Raccolta di {num_immagini} immagini dal WebDataset...")
    for sample in data_iter:
        for img in sample[0]:
            if len(immagini_raccolte) < num_immagini:
                immagini_raccolte.append(img.cpu())
            else:
                break
        if len(immagini_raccolte) >= num_immagini:
            break
            
    if len(immagini_raccolte) == 0:
        print("Errore: Il dataset è vuoto o non è stato possibile estrarre immagini.")
        return

    # Se lo stream si è esaurito prima del previsto, avvisiamo l'utente
    if len(immagini_raccolte) < num_immagini:
        print(f"Nota: Trovate solo {len(immagini_raccolte)} immagini rispetto alle {num_immagini} richieste.")

    # 3. Stack delle immagini individuali in un unico batch tensor [B, C, H, W]
    immagini = torch.stack(immagini_raccolte, dim=0)
    print("Dimensione del batch di immagini raccolte:", immagini.size())

    # 4. Denormalizzazione (opzionale ma consigliata se usi transforms.Normalize)
    if mean is not None and std is not None:
        # Convertiamo in tensor con dimensioni compatibili [1, C, 1, 1] per il broadcasting su un batch
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)
        # Ripristiniamo i colori originali: img * std + mean
        immagini = immagini * std_t + mean_t
    #for the first image in immagini get the max and min value and print them
    print(f"Valori pixel prima del clamp: min={immagini.min().item():.4f}, max={immagini.max().item():.4f}")
    
    # Assicuriamoci che i valori siano nel range [0, 1] per il salvataggio corretto
    immagini = torch.clamp(immagini, 0.0, 1.0)

    # 5. Creiamo la cartella di destinazione se non esiste
    cartella = os.path.dirname(output_path)
    if cartella and not os.path.exists(cartella):
        os.makedirs(cartella)

    # 6. Creiamo la griglia e salviamo su file
    nrow = int(len(immagini_raccolte) ** 0.5)
    vutils.save_image(immagini, output_path, nrow=nrow, padding=2, normalize=False)
    
    print(f"Anteprima salvata con successo in: {output_path}")