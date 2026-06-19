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

def debug_images_PD(mean,std,loader,out_dir):
    def _denorm_to_hwc(img, mean, std):
        """(C,H,W) tensor -> (H,W[,C]) numpy in [0,1] for imshow."""
        img = img.detach().cpu().float()
        if mean is not None and std is not None:
            m = torch.tensor(mean).view(-1, 1, 1)
            s = torch.tensor(std).view(-1, 1, 1)
            img = img * s + m
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return img[:, :, 0] if img.shape[-1] == 1 else img      # squeeze grayscale


    def debug_show_batch(batch, out_dir="debug_batches",
                        subject_ids=None, slot_to_q=None,
                        mean=mean, std=std,
                        max_subjects=None, dpi=110):
        """
        Save one PNG per subject. Layout: rows = views (k), cols = questionnaires (slot order).

        batch : (frames, seq_ids, slot_ids, lengths, labels) from collate_variable_sequences
            frames   (N, k, C, H, W)   N = sum(T_i)
            seq_ids  (N,)              subject index 0..B-1 of each frame
            slot_ids (N,)              questionnaire slot of each frame
        subject_ids : optional list[str] length B (else "subject_{b}")
        slot_to_q   : optional dict/callable slot->label (else "q{slot+1}")
        mean,std    : denormalization; pass None, None if frames already in [0,1]
        """
        frames, seq_ids, slot_ids, lengths, labels = batch
        seq_ids, slot_ids = seq_ids.cpu(), slot_ids.cpu()
        B = lengths.size(0)

        if slot_to_q is None:
            slot_name = lambda s: f"q{s + 1}"
        elif callable(slot_to_q):
            slot_name = slot_to_q
        else:
            slot_name = lambda s: slot_to_q.get(s, f"q{s + 1}")

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        n_show = B if max_subjects is None else min(B, max_subjects)

        for b in range(n_show):
            sel = (seq_ids == b).nonzero(as_tuple=True)[0]     # frame indices for subject b
            if sel.numel() == 0:
                continue
            sel = sel[torch.argsort(slot_ids[sel])]            # questionnaires in slot order
            slots = slot_ids[sel].tolist()

            sub = frames[sel]                                  # (T_b, k, C, H, W)
            T_b, k, C = sub.shape[:3]

            fig, axes = plt.subplots(k, T_b, figsize=(2.2 * T_b, 2.2 * k), squeeze=False)
            for j in range(T_b):                               # column = questionnaire
                axes[0, j].set_title(slot_name(slots[j]), fontsize=10)
                for i in range(k):                             # row = view
                    ax = axes[i, j]
                    ax.imshow(_denorm_to_hwc(sub[j, i], mean, std),
                            cmap="gray" if C == 1 else None)
                    ax.set_xticks([]); ax.set_yticks([])
                    if j == 0:
                        ax.set_ylabel(f"view {i}", fontsize=9)

            sid = subject_ids[b] if subject_ids is not None else f"subject_{b}"
            lab = labels[b].item() if torch.is_tensor(labels) else labels[b]
            fig.suptitle(f"{sid}   label={lab}   T={T_b}  k={k}", fontsize=11)
            fig.tight_layout()

            path = os.path.join(out_dir, f"{sid}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

        return paths
    batch = next(iter(loader))                  # WebLoader(batch_size=None)
    os.makedirs(out_dir, exist_ok=True)
    debug_show_batch(batch, out_dir=out_dir)

def debug_print_batch_meta(batch, subject_ids=None, slot_to_q=None, max_subjects=None):
    """
    Print all non-image elements of a batch from collate_variable_sequences.

    batch : (frames, seq_ids, slot_ids, lengths, labels[, subject_ids])
        frames is skipped; everything else is printed.
    subject_ids : optional list[str] (used if batch doesn't already carry it)
    slot_to_q   : optional dict/callable slot->label (else "q{slot+1}")
    """
    # unpack, tolerating the 5- or 6-element variant
    frames, seq_ids, slot_ids, lengths, labels = batch[:5]
    if subject_ids is None and len(batch) >= 6:
        subject_ids = batch[5]

    seq_ids  = seq_ids.cpu()
    slot_ids = slot_ids.cpu()
    lengths  = lengths.cpu()
    B = lengths.size(0)
    N = seq_ids.size(0)

    if slot_to_q is None:
        slot_name = lambda s: f"q{s + 1}"
    elif callable(slot_to_q):
        slot_name = slot_to_q
    else:
        slot_name = lambda s: slot_to_q.get(s, f"q{s + 1}")

    print("=" * 60)
    print(f"BATCH META   B={B} subjects   N={N} frames   "
          f"frames.shape={tuple(frames.shape)}")
    print("-" * 60)

    # raw collated arrays (everything that isn't the image tensor)
    print(f"lengths   ({tuple(lengths.shape)}): {lengths.tolist()}")
    print(f"labels    ({tuple(labels.shape)}): "
          f"{labels.tolist() if torch.is_tensor(labels) else labels}")
    print(f"seq_ids   ({tuple(seq_ids.shape)}): {seq_ids.tolist()}")
    print(f"slot_ids  ({tuple(slot_ids.shape)}): {slot_ids.tolist()}")
    if subject_ids is not None:
        print(f"subject_ids ({len(subject_ids)}): {list(subject_ids)}")

    # consistency check: lengths must match the frame counts implied by seq_ids
    counts = torch.bincount(seq_ids, minlength=B)
    mismatch = (counts != lengths).nonzero(as_tuple=True)[0].tolist()
    print("-" * 60)
    print(f"frames per subject (from seq_ids): {counts.tolist()}")
    if mismatch:
        print(f"  ** WARNING: lengths != seq_id counts at subjects {mismatch} **")
    else:
        print("  lengths consistent with seq_ids ✓")

    # per-subject breakdown
    print("-" * 60)
    n_show = B if max_subjects is None else min(B, max_subjects)
    for b in range(n_show):
        sel = (seq_ids == b).nonzero(as_tuple=True)[0]
        sel = sel[torch.argsort(slot_ids[sel])]            # slot order
        slots = slot_ids[sel].tolist()
        qs    = [slot_name(s) for s in slots]

        sid = subject_ids[b] if subject_ids is not None else f"subject_{b}"
        lab = labels[b].item() if torch.is_tensor(labels) else labels[b]

        dup = len(slots) != len(set(slots))
        print(f"[{b}] {sid}  label={lab}  T={len(slots)}  "
              f"slots={slots}  questionnaires={qs}"
              + ("   ** DUPLICATE SLOT **" if dup else ""))
    print("=" * 60)