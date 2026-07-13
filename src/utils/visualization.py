import os
import torch
import torchvision.utils as vutils
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import gradio as gr
from matplotlib.figure import Figure   # NOT pyplot — no display needed
from matplotlib.backends.backend_agg import FigureCanvasAgg
import traceback

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

# Visualization for PD time series data
def debug_images_PD(mean,std,loader,out_dir,input_is_batch=False): #(N, k, C, H, W), N = sum(T_i) - Format
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
        frames, seq_ids, slot_ids, lengths, labels, resizing_factors,subject_ids, modalities = batch
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
            modes = [modalities[i] for i in sel.tolist()]

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
                        ax.set_ylabel(f"{modes[j][i]}", fontsize=9)

            sid = subject_ids[b] if subject_ids is not None else f"subject_{b}"
            lab = labels[b].item() if torch.is_tensor(labels) else labels[b]
            fig.suptitle(f"{sid}   label={lab}   T={T_b}  k={k}", fontsize=11)
            fig.tight_layout()

            path = os.path.join(out_dir, f"subject_{b}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

        return paths
    if input_is_batch:
        batch = loader
    else:
        batch = next(iter(loader))                  # WebLoader(batch_size=None)
    os.makedirs(out_dir, exist_ok=True)
    debug_show_batch(batch, out_dir=out_dir)
def debug_images_PD_with_meta(mean, std, batch, out_dir,
                              slot_to_q=None,
                              max_subjects=None, dpi=110,
                              meta_fontsize=5.5,
                              img_h=2.4, meta_h=2.8,
                              compact=False):
    """
    Same idea as debug_images_PD, but prints a metadata panel directly UNDER
    every image of the timeseries.

    Layout, one PNG per subject:
        columns = questionnaires / timepoints (slot order)  -> the timeseries
        rows    = views (k); each view takes TWO physical rows:
                    - image row
                    - metadata row (tensor_debug_info for that exact frame)

    So for each subject you get a grid: one line (row) per view, the line being a
    timeseries of images, with the metadata for each image shown right below it.

    Params
    ------
    mean, std     : denormalization (pass None, None if frames already in [0,1])
    slot_to_q     : optional dict or callable slot -> label (else "q{slot+1}")
    meta_fontsize : font size of the metadata text (small, since cells are tiny)
    img_h, meta_h : relative heights of the image row vs the metadata row
    compact       : if True, use a short summary instead of full tensor_debug_info
    """

    # ---- denorm helper (unchanged) -------------------------------------
    def _denorm_to_hwc(img, mean, std):
        """(C,H,W) tensor -> (H,W[,C]) numpy in [0,1] for imshow."""
        img = img.detach().cpu().float()
        if mean is not None and std is not None:
            m = torch.tensor(mean).view(-1, 1, 1)
            s = torch.tensor(std).view(-1, 1, 1)
            img = img * s + m
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return img[:, :, 0] if img.shape[-1] == 1 else img

    # ---- compact fallback metadata -------------------------------------
    def _compact_meta(t, name="tensor", norm_mean=None, norm_std=None):
        tc = t.detach().cpu().float()
        vmin, vmax = tc.min().item(), tc.max().item()
        out = (f"{name}\n"
               f"shape {tuple(t.shape)}\n"
               f"min {vmin:+.3f}  max {vmax:+.3f}\n"
               f"mean {tc.mean().item():+.3f}  std {tc.std().item():+.3f}")
        if norm_mean is not None and norm_std is not None and tc.ndim == 3:
            m = torch.as_tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1)
            s = torch.as_tensor(norm_std,  dtype=torch.float32).view(-1, 1, 1)
            if m.shape[0] == tc.shape[0]:
                d = tc * s + m
                out += f"\ndenorm [{d.min().item():+.3f}, {d.max().item():+.3f}]"
        return out

    # choose metadata generator
    if compact:
        meta_fn = _compact_meta
    else:
        try:
            meta_fn = tensor_debug_info          # your full diagnostic string
        except NameError:
            meta_fn = _compact_meta              # graceful fallback

    # ---- slot label resolver (kept from original) ----------------------
    if slot_to_q is None:
        slot_name = lambda s: f"q{s + 1}"
    elif callable(slot_to_q):
        slot_name = slot_to_q
    else:
        slot_name = lambda s: slot_to_q.get(s, f"q{s + 1}")

    def debug_show_batch(batch, out_dir):
        frames, seq_ids, slot_ids, lengths, labels, \
            resizing_factors, subject_ids, modalities = batch
        seq_ids, slot_ids = seq_ids.cpu(), slot_ids.cpu()
        B = lengths.size(0)

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        n_show = B if max_subjects is None else min(B, max_subjects)

        for b in range(n_show):
            sel = (seq_ids == b).nonzero(as_tuple=True)[0]   # frames of subject b
            if sel.numel() == 0:
                continue
            sel = sel[torch.argsort(slot_ids[sel])]          # slot order
            slots = slot_ids[sel].tolist()
            modes = [modalities[i] for i in sel.tolist()]
            sub = frames[sel]                                # (T_b, k, C, H, W)
            T_b, k, C = sub.shape[:3]

            # 2 physical rows per view: [image row, metadata row]
            nrows = 2 * k
            fig, axes = plt.subplots(
                nrows, T_b,
                figsize=(2.6 * T_b, (img_h + meta_h) * k),
                squeeze=False,
                gridspec_kw={"height_ratios": [img_h, meta_h] * k},
            )

            for j in range(T_b):                             # column = timepoint
                axes[0, j].set_title(slot_name(slots[j]), fontsize=10)
                for i in range(k):                           # view index
                    r_img = 2 * i
                    r_txt = 2 * i + 1

                    # --- image ---
                    ax = axes[r_img, j]
                    ax.imshow(_denorm_to_hwc(sub[j, i], mean, std),
                              cmap="gray" if C == 1 else None)
                    ax.set_xticks([]); ax.set_yticks([])
                    if j == 0:
                        ax.set_ylabel(f"{modes[j][i]}", fontsize=9)

                    # --- metadata panel directly under that image ---
                    tax = axes[r_txt, j]
                    tax.axis("off")
                    name = f"{slot_name(slots[j])} | {modes[j][i]}"
                    txt = meta_fn(sub[j, i], name=name,
                                  norm_mean=mean, norm_std=std)
                    tax.text(0.0, 1.0, txt, transform=tax.transAxes,
                             va="top", ha="left", family="monospace",
                             fontsize=meta_fontsize, linespacing=1.05)

            sid = subject_ids[b] if subject_ids is not None else f"subject_{b}"
            lab = labels[b].item() if torch.is_tensor(labels) else labels[b]
            fig.suptitle(f"{sid}   label={lab}   T={T_b}  k={k}", fontsize=11)
            fig.tight_layout()
            path = os.path.join(out_dir, f"subject_{b}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return paths

    os.makedirs(out_dir, exist_ok=True)
    return debug_show_batch(batch, out_dir=out_dir)

def debug_print_batch_meta(batch, subject_ids=None, slot_to_q=None, max_subjects=None):
    """
    Print all non-image elements of a batch from collate_variable_sequences.

    batch : (frames, seq_ids, slot_ids, lengths, labels, resizing_factors, subjects)
        frames is skipped; everything else is printed.
    subject_ids : optional list[str] (used if batch doesn't already carry it)
    slot_to_q   : optional dict/callable slot->label (else "q{slot+1}")
    """
    # unpack, tolerating the 5- or 6-element variant
    frames, seq_ids, slot_ids, lengths, labels, resizing_factors,subject_ids, modalities = batch

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
    print(f"resizing_factors ({len(resizing_factors)}): {resizing_factors}")
    print(f"subject_ids ({len(subject_ids)}): {list(subject_ids)}")
    print(f"modalities ({len(modalities)}): {modalities}")

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

# Show images with info
def save_img_with_info(image_data,properties_text,path):
    # Create a 1-row, 2-column figure layout
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.5, 1]})

    # Column 1: Display the actual image
    axs[0].imshow(image_data, cmap='viridis')
    axs[0].set_title("Processed Sample", fontsize=14, fontweight='bold')
    axs[0].axis('off')  # Hide image axis ticks

    # Column 2: Turn off the plot lines and render the long text block
    axs[1].axis('off')
    axs[1].text(
        x=0.0, y=1.0,               # Coordinate starting point (top-left of this subplot)
        s=properties_text, 
        fontsize=11, 
        fontfamily='monospace',     # Monospace keeps alignment neat
        verticalalignment='top', 
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='gray', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
def save_img_with_info_views(image_list, text_properties_list, path):
    n = len(image_list)
    # n rows, 2 columns. squeeze=False keeps axs 2D even when n == 1
    fig, axs = plt.subplots(
        n, 2,
        figsize=(12, 6 * n),
        gridspec_kw={'width_ratios': [1.5, 1]},
        squeeze=False
    )

    for i, (image_data, properties_text) in enumerate(zip(image_list, text_properties_list)):
        # Column 1: Display the actual image
        axs[i][0].imshow(image_data)
        axs[i][0].set_title("Processed Sample", fontsize=14, fontweight='bold')
        axs[i][0].axis('off')  # Hide image axis ticks

        # Column 2: Turn off the plot lines and render the long text block
        axs[i][1].axis('off')
        axs[i][1].text(
            x=0.0, y=1.0,               # Coordinate starting point (top-left of this subplot)
            s=properties_text,
            fontsize=11,
            fontfamily='monospace',     # Monospace keeps alignment neat
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='gray', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # avoid keeping figures open in memory across calls


#Debug info on images
def tensor_debug_info(t, name="tensor", norm_mean=None, norm_std=None):
    """Build a human-readable diagnostic string from a raw image tensor (C×H×W or H×W).

    If norm_mean/norm_std are provided, also reports whether denormalizing
    recovers a valid [0,1] display range — turning the 'guess' into a check.
    """
    t_cpu = t.detach().cpu().float()
    lines = []

    # --- shape & dtype ---
    lines.append(f"{name}")
    lines.append(f"{'shape':<14}: {tuple(t.shape)}")
    lines.append(f"{'dtype':<14}: {t.dtype}")
    lines.append(f"{'device':<14}: {t.device}")

    # --- value range (the key normalization clue) ---
    vmin, vmax = t_cpu.min().item(), t_cpu.max().item()
    lines.append(f"{'min':<14}: {vmin:+.4f}")
    lines.append(f"{'max':<14}: {vmax:+.4f}")
    lines.append(f"{'mean':<14}: {t_cpu.mean().item():+.4f}")
    lines.append(f"{'std':<14}: {t_cpu.std().item():+.4f}")

    # --- per-channel stats (catches uneven normalization) ---
    if t_cpu.ndim == 3 and t_cpu.shape[0] in (1, 3, 4):
        for c in range(t_cpu.shape[0]):
            ch = t_cpu[c]
            lines.append(
                f"  ch{c} mean/std : {ch.mean().item():+.3f} / {ch.std().item():+.3f}"
                f"  [{ch.min().item():+.3f}, {ch.max().item():+.3f}]"
            )

    # --- interpretation heuristics ---
    if vmin < -0.01 and vmax > 1.01:
        guess = "likely NORMALIZED (mean/std) — denorm before display"
    elif 0.0 <= vmin and vmax <= 1.01:
        guess = "looks like [0,1] float — ToPILImage OK"
    elif vmax > 1.5 and vmax <= 255.5:
        guess = "looks like [0,255] range"
    else:
        guess = "unusual range — inspect manually"
    lines.append(f"{'guess':<14}: {guess}")

    # --- denorm check (confirmation when mean/std are known) ---
    if norm_mean is not None and norm_std is not None and t_cpu.ndim == 3:
        mean = torch.as_tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1)
        std  = torch.as_tensor(norm_std,  dtype=torch.float32).view(-1, 1, 1)
        if mean.shape[0] == t_cpu.shape[0]:        # channel count must match
            denorm = t_cpu * std + mean
            dmin, dmax = denorm.min().item(), denorm.max().item()
            lines.append(f"{'denorm range':<14}: [{dmin:+.4f}, {dmax:+.4f}]")
            # small tolerance for float drift / mild clipping at edges
            if -0.02 <= dmin and dmax <= 1.02:
                verdict = "OK — denorm recovers [0,1]"
            else:
                spill_lo = max(0.0, -dmin)
                spill_hi = max(0.0, dmax - 1.0)
                verdict = (f"OUT OF RANGE by ({spill_lo:.3f} low, {spill_hi:.3f} high) "
                           f"— wrong mean/std, or tensor isn't normalized")
            lines.append(f"{'denorm check':<14}: {verdict}")
        else:
            lines.append(f"{'denorm check':<14}: SKIPPED — mean/std has "
                         f"{mean.shape[0]} ch, tensor has {t_cpu.shape[0]}")

    # --- health flags ---
    flags = []
    if torch.isnan(t_cpu).any(): flags.append("NaN present!")
    if torch.isinf(t_cpu).any(): flags.append("Inf present!")
    if vmin == vmax:             flags.append("constant tensor (all same value)")
    if flags:
        lines.append(f"{'flags':<14}: " + ", ".join(flags))

    return "\n".join(lines)


# Interactive visualization PD
def _denorm_to_hwc(img, mean, std):
    """(C,H,W) tensor -> (H,W[,C]) numpy in [0,1] for imshow."""
    img = img.detach().cpu().float()
    if mean is not None and std is not None:
        m = torch.tensor(mean).view(-1, 1, 1)
        s = torch.tensor(std).view(-1, 1, 1)
        img = img * s + m
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img[:, :, 0] if img.shape[-1] == 1 else img


def iter_subjects(batch):
    """Yield one dict per subject from a single collated batch."""
    (frames, seq_ids, slot_ids, lengths, labels,
     resizing_factors, subject_ids, modalities) = batch
    seq_ids, slot_ids = seq_ids.cpu(), slot_ids.cpu()
    B = lengths.size(0)
    for b in range(B):
        sel = (seq_ids == b).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        sel = sel[torch.argsort(slot_ids[sel])]          # slot order
        yield dict(
            subject_id=(subject_ids[b] if subject_ids is not None else f"subject_{b}"),
            label=(labels[b].item() if torch.is_tensor(labels) else labels[b]),
            slots=slot_ids[sel].tolist(),
            modes=[modalities[i] for i in sel.tolist()],
            frames=frames[sel],                          # (T_b, k, C, H, W)
        )


def subject_stream(loader, max_batches=None):
    """Lazy generator over all subjects, pulling batches on demand."""
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        yield from iter_subjects(batch)


def _format_meta(meta):
    if meta is None:
        return "(no metadata)"
    if isinstance(meta, dict):
        return "\n".join(f"{k}: {v}" for k, v in meta.items())
    return str(meta)


class SubjectViewer:
    """
    Interactive per-subject viewer.
      right / space / n : next     left / p : previous     q / esc : quit
    get_meta(subject_id) -> dict|str is called each time a subject is shown.
    """
    def __init__(self, loader, mean, std, get_meta=None, max_batches=None,
                 slot_to_q=None, figsize=(13, 7)):
        self.gen = subject_stream(loader, max_batches)
        self.cache, self.idx = [], -1
        self.mean, self.std = mean, std
        self.get_meta = get_meta or (lambda sid: None)

        if slot_to_q is None:
            self.slot_name = lambda s: f"q{s + 1}"
        elif callable(slot_to_q):
            self.slot_name = slot_to_q
        else:
            self.slot_name = lambda s: slot_to_q.get(s, f"q{s + 1}")

        self.fig = plt.figure(figsize=figsize)
        self.content_sf, ctrl_sf = self.fig.subfigures(2, 1, height_ratios=[12, 1])
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # persistent buttons (survive content redraws)
        b_prev, b_next, b_quit = ctrl_sf.subplots(1, 3)
        self._bp = Button(b_prev, "◀ Prev");  self._bp.on_clicked(lambda e: self.prev())
        self._bn = Button(b_next, "Next ▶");  self._bn.on_clicked(lambda e: self.next())
        self._bq = Button(b_quit, "Quit");    self._bq.on_clicked(lambda e: plt.close(self.fig))

        self.next()  # show first subject

    def _advance(self):
        if self.idx + 1 < len(self.cache):
            self.idx += 1
            return True
        try:
            self.cache.append(next(self.gen))
        except StopIteration:
            return False
        self.idx += 1
        return True

    def next(self):
        if self._advance():
            self._render()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._render()

    def _on_key(self, event):
        if event.key in (" ", "right", "n"):
            self.next()
        elif event.key in ("left", "p"):
            self.prev()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _render(self):
        rec = self.cache[self.idx]
        frames = rec["frames"]
        T_b, k, C = frames.shape[:3]

        sf = self.content_sf
        sf.clear()
        gs = sf.add_gridspec(k, T_b + 1, width_ratios=[1] * T_b + [1.3])

        for j in range(T_b):
            for i in range(k):
                ax = sf.add_subplot(gs[i, j])
                ax.imshow(_denorm_to_hwc(frames[j, i], self.mean, self.std),
                          cmap="gray" if C == 1 else None)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(self.slot_name(rec["slots"][j]), fontsize=10)
                if j == 0:
                    ax.set_ylabel(str(rec["modes"][j][i]), fontsize=9)

        axm = sf.add_subplot(gs[:, -1]); axm.axis("off")
        axm.text(0, 1,
                 f"[{self.idx + 1}] {rec['subject_id']}\n\n{_format_meta(self.get_meta(rec['subject_id']))}",
                 va="top", ha="left", fontsize=9, family="monospace", wrap=True)

        sf.suptitle(f"{rec['subject_id']}   label={rec['label']}   T={T_b}  k={k}",
                    fontsize=11)
        self.fig.canvas.draw_idle()

# Gradio interactive visualization
def _fig_to_array(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3]   # RGB, drop alpha

def _meta_markdown(meta):
    if isinstance(meta, dict):
        return "\n".join(f"| **{k}** | {v} |" for k, v in meta.items())
    return str(meta)


class SubjectApp:
    def __init__(self, loader, mean, std, get_meta, max_batches=None, slot_to_q=None):
        self.mean, self.std, self.get_meta = mean, std, get_meta
        if slot_to_q is None:
            self.slot_name = lambda s: f"q{s + 1}"
        elif callable(slot_to_q):
            self.slot_name = slot_to_q
        else:
            self.slot_name = lambda s: slot_to_q.get(s, f"q{s + 1}")

        # --- materialize everything NOW, in the main thread ---
        print("[init] extracting subjects...", flush=True)
        self.cache = []
        for rec in subject_stream(loader, max_batches):
            rec["frames"] = rec["frames"].detach().cpu()   # off GPU, safe to hold
            self.cache.append(rec)
        print(f"[init] cached {len(self.cache)} subjects", flush=True)
        self.idx = -1

        self.debug_dir = "/home/a_morelli/vscode_projects/model_training/data/dataset_info_PD/random_samples/interactive"    # somewhere you can ls / scp from
        os.makedirs(self.debug_dir, exist_ok=True)


    def _render(self):
        rec = self.cache[self.idx]
        frames = rec["frames"]
        T_b, k, C = frames.shape[:3]

        # sanity print -> lands in viewer_<jobid>.log
        raw = frames.detach().cpu().float()
        print(f"[render] idx={self.idx} id={rec['subject_id']} "
              f"shape={tuple(frames.shape)} raw_min={raw.min():.3f} raw_max={raw.max():.3f}",
              flush=True)

        fig = Figure(figsize=(2.2 * T_b, 2.2 * k))
        axes = fig.subplots(k, T_b, squeeze=False)
        for j in range(T_b):
            axes[0, j].set_title(self.slot_name(rec["slots"][j]), fontsize=10)
            for i in range(k):
                ax = axes[i, j]
                arr = _denorm_to_hwc(frames[j, i], self.mean, self.std)
                if i == 0 and j == 0:
                    print(f"[render] denormed min={arr.min():.3f} max={arr.max():.3f}", flush=True)
                ax.imshow(arr, cmap="gray" if C == 1 else None)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(str(rec["modes"][j][i]), fontsize=9)
        fig.suptitle(f"{rec['subject_id']}  label={rec['label']}  T={T_b} k={k}")
        fig.tight_layout()

        path = os.path.join(self.debug_dir, f"current_{self.idx}.png")
        fig.savefig(path, dpi=110)                     # ground truth on disk

        header = f"### [{self.idx + 1}] `{rec['subject_id']}` — label {rec['label']}"
        meta_md = "| field | value |\n|---|---|\n" + _meta_markdown(self.get_meta(rec["subject_id"]))
        return path, header, meta_md                   # filepath -> gr.Image

    def next(self):
        try:
            if self.idx + 1 < len(self.cache):
                self.idx += 1
            return self._render()
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            return None, "### ERROR in next()", f"```\n{tb}\n```"

    def prev(self):
        try:
            if self.idx > 0:
                self.idx -= 1
            return self._render()
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            return None, "### ERROR in prev()", f"```\n{tb}\n```"


def launch_interactive_PD(loader, mean, std, get_meta, max_batches=None, port=7860):
    app = SubjectApp(loader, mean, std, get_meta, max_batches)
    with gr.Blocks() as demo:
        header = gr.Markdown()
        with gr.Row():
            plot = gr.Image(label="subject", type="filepath")
            meta = gr.Markdown()
        with gr.Row():
            b_prev = gr.Button("◀ Prev")
            b_next = gr.Button("Next ▶")
        b_next.click(app.next, outputs=[plot, header, meta])
        b_prev.click(app.prev, outputs=[plot, header, meta])
        demo.load(app.next, outputs=[plot, header, meta])   # show first on open
    demo.launch(server_name="0.0.0.0", server_port=port)     # bind all interfaces