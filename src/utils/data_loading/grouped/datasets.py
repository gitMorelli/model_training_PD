import torch
from src.utils.data_loading.pd_dataloaders import _collate_subject_level


def collate_groups_PD(batch_of_groups, debug=False):
    sequences, labels, subject_ids = [], [], []
    questionnaires, modalities_list, resized_list = [], [], []
    group_ids = []

    for g_idx, (gid, group) in enumerate(batch_of_groups):
        for seq, lab, sid, qs, mods, res in group:
            sequences.append(seq)
            labels.append(lab)
            subject_ids.append(sid)
            questionnaires.append(qs)
            modalities_list.append(mods)
            resized_list.append(res)
            group_ids.append(g_idx)

    labels = torch.stack(labels)
    out = _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                 modalities_list, resized_list, debug=debug)
    return (*out, torch.tensor(group_ids))

def collate_variable_sequences_PD_grouped(samples, debug=False):
    """Collate for the grouped loader.
 
    Each sample is one GROUP:
        (sequences, labels, subjects, questionnaires, modalities, resized, group_id)
    where sequences is (G, T, k, C, H, W) if stacked, or a list of per-subject
    sequences (debug mode, or if you switched to list-yield for variable T).
 
    Groups are flattened into individual subjects, then collated with the same
    core as the flat version. Two extra outputs are appended:
        group_ids     : list of len N, the group_id string for each subject
        group_seq_ids : tensor (N,), index of the group within this batch
                        (lets you regroup subjects downstream)
    """
    sequences, labels_parts, subject_ids = [], [], []
    questionnaires, modalities_list, resized_list = [], [], []
    group_ids, group_seq_ids = [], []
 
    for g, s in enumerate(samples):
        (grp_seqs, grp_labels, grp_subjects,
         grp_qs, grp_mods, grp_res, group_id) = s
 
        # unstack (G, T, k, C, H, W) -> list of G tensors (T, k, C, H, W)
        if isinstance(grp_seqs, torch.Tensor):
            grp_seqs = list(torch.unbind(grp_seqs, dim=0))
 
        sequences.extend(grp_seqs)
        labels_parts.append(grp_labels)                    # tensor (G,)
        subject_ids.extend(grp_subjects)
        questionnaires.extend(grp_qs)
        modalities_list.extend(grp_mods)
        resized_list.extend(grp_res)
        group_ids.extend([group_id] * len(grp_subjects))
        group_seq_ids.extend([g] * len(grp_subjects))
 
    labels = torch.cat(labels_parts)
 
    base = _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                  modalities_list, resized_list, debug=debug)
 
    return (*base, group_ids, torch.tensor(group_seq_ids))
