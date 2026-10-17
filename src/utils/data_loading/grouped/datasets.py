import torch
from src.utils.data_loading.pd_shared_helpers import _make_subject_sequence_builder, _parse_sample_json, _index_images
from src.utils.data_loading.pd_shared_helpers import _collate_subject_level


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

def create_sequence_group_flattener_PD_multiview(transform_func, augmentation_transform_list,
                                    modality_string_list, original_modality_names,
                                    exclusion_set, grid_dict, censor_time,
                                    filter_modality, exp_params=None,huggingface_transform=False,
                                    invert_color=False, debug=False, train_df=None, is_val=False, **kw):
    build_sequence = _make_subject_sequence_builder(
        transform_func, augmentation_transform_list, modality_string_list,
        original_modality_names, grid_dict, censor_time, filter_modality,
        huggingface_transform, invert_color, debug, train_df,exp_params=exp_params,
        is_val=is_val)

    def flatten_groups(src):
        for sample in src:
            json_data = _parse_sample_json(sample)
            if json_data is None:
                continue

            image_pool = _index_images(sample, grouped=True)  # subj -> q -> mod
            group_id = json_data["group_id"]

            '''print("Start debugging dataloader ...")
            print(image_pool.keys())
            print("length exclusion set:", len(exclusion_set))
            print("json -> \n", json_data["subjects"], flush=True)
            assert 1==0, "debug"'''

            group = []
            for subject_id, meta in json_data["subjects"].items():
                if subject_id in exclusion_set:
                    continue

                rempli_seulq12 = 1  #i ignore the rempli_seul problem for now
                result = build_sequence(subject_id, meta["last_q"],
                                        meta["questionnaire_info"],
                                        image_pool.get(subject_id, {}),meta['case_grid_pattern'], rempli_seulq12)
                if result is None:
                    continue
                sequence, questionnaires, modalities, resized, _, _ = result
                label = torch.tensor(meta["label"], dtype=torch.long)
                group.append((sequence, label, subject_id,
                              questionnaires, modalities, resized))

            # only yield valid groups: exactly one positive, ≥1 control
            labels = [g[1].item() for g in group]
            if sum(labels) == 1 and len(group) >= 2:
                yield group_id, group

    return flatten_groups
