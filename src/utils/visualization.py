import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def debug_images_dataset(dataset, output_path="anteprima_dataset.png", num_immagini=16, mean=None, std=None, n_stacked=1):
    """
    Estrae un numero specifico di immagini da un WebDataset (o standard Dataset) 
    iterando sugli elementi e le salva in una griglia su file.
    """
    if n_stacked > 1:
        debug_images_dataset_stacked(dataset, output_path, num_immagini, mean, std, n_stacked)
        return

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

def debug_images_dataset_stacked(
    dataset, 
    output_path="anteprima_dataset.png", 
    num_immagini=16, 
    mean=None, 
    std=None, 
    n_stacked=3 # Nomi ordinati delle modalità stacked
):
    """
    Estrae un numero specifico di campioni multimodali da un WebDataset,
    separa le 3 modalità stacked e salva una griglia distinta per ciascuna di esse.
    """
    # 1. Creiamo un DataLoader temporaneo (batch_size=None mantiene lo streaming nativo)
    dataloader = DataLoader(
        dataset, 
        num_workers=0, 
        batch_size=None, 
        prefetch_factor=None,
    )
    
    modality_names = [str(i) for i in range(n_stacked)]  # Esempio: ['0', '1', '2'] per 3 modalità

    # Inizializziamo un dizionario di liste, una per ogni modalità
    raccolte_per_mod = {name: [] for name in modality_names}
    campioni_raccolti = 0
    data_iter = iter(dataloader)
    
    print(f"Raccolta di {num_immagini} campioni multimodali dal WebDataset...")
    
    # 2. Iteriamo ed estraiamo i campioni
    for sample in data_iter:
        print('dim in debug images: ', sample[0].shape)
        x = sample[0] # Forma attesa per singolo campione: [3, 3, H, W]
        
        # Se x è 4D (singolo), lo trasformiamo in un batch fake di dimensione 1: [1, 3, 3, H, W]
        # Se è già 5D, lo lasciamo così com'è.
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        # Ora x è GARANTITO essere 5D: [Batch, Modalità, Canali, H, W]
        # Iteriamo lungo la reale dimensione del batch corrente
        for b in range(x.size(0)):
            if campioni_raccolti < num_immagini:
                for i, name in enumerate(modality_names):
                    # Estraiamo la singola immagine pulita a 3 dimensioni [Canali, H, W]
                    raccolte_per_mod[name].append(x[b, i].cpu())
                campioni_raccolti += 1
            else:
                break
        if campioni_raccolti >= num_immagini:
            break
            
    if campioni_raccolti == 0:
        print("Errore: Il dataset è vuoto o non è stato possibile estrarre immagini.")
        return

    if campioni_raccolti < num_immagini:
        print(f"Nota: Trovati solo {campioni_raccolti} campioni rispetto ai {num_immagini} richiesti.")

    # 3. Creiamo la cartella di destinazione se non esiste
    cartella = os.path.dirname(output_path)
    if cartella and not os.path.exists(cartella):
        os.makedirs(cartella)

    # Separiamo il nome del file dall'estensione per poter inserire il suffisso della modalità
    base_path, ext = os.path.splitext(output_path)

    # 4. Elaboriamo e salviamo una griglia indipendente per ogni modalità
    for name in modality_names:
        lista_immagini = raccolte_per_mod[name]
        
        # Stack delle immagini individuali in un unico batch tensor [B, C, H, W]
        immagini = torch.stack(lista_immagini, dim=0)
        print(f"\n--- Elaborazione modalità: {name.upper()} ---")
        print(f"Dimensione del batch per {name}:", immagini.size())

        # 5. Denormalizzazione (opzionale)
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(1, -1, 1, 1)
            std_t = torch.tensor(std).view(1, -1, 1, 1)
            immagini = immagini * std_t + mean_t
            
        print(f"Valori pixel prima del clamp ({name}): min={immagini.min().item():.4f}, max={immagini.max().item():.4f}")
        
        # Assicuriamoci che i valori siano nel range [0, 1]
        immagini = torch.clamp(immagini, 0.0, 1.0)

        # 6. Generazione del percorso specifico per la modalità corrente
        # Esempio: "anteprima_dataset.png" -> "anteprima_dataset_digits.png"
        modality_output_path = f"{base_path}_{name}{ext}"

        # 7. Creazione della griglia e salvataggio su file
        nrow = int(campioni_raccolti ** 0.5)
        vutils.save_image(immagini, modality_output_path, nrow=nrow, padding=2, normalize=False)
        
        print(f"Anteprima [{name}] salvata con successo in: {modality_output_path}")
