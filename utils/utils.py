import re

import torch
from utils.constants import GRAPH_TOKEN_INDEX, GRAPH_TOKEN_TO_INDEX


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def tokenizer_graph_token(prompt, tokenizer, graph_token_index=GRAPH_TOKEN_INDEX, return_tensors=None):
    token_map = dict(GRAPH_TOKEN_TO_INDEX)
    token_map["<graph>"] = graph_token_index
    pattern = "(" + "|".join(re.escape(token) for token in token_map) + ")"
    pieces = re.split(pattern, prompt)
    input_ids = []
    strip_bos = False
    for piece in pieces:
        if piece in token_map:
            input_ids.append(token_map[piece])
            continue
        if not piece:
            continue
        ids = tokenizer(piece).input_ids
        if not input_ids and ids and ids[0] == tokenizer.bos_token_id:
            input_ids.append(ids[0])
            strip_bos = True
            ids = ids[1:]
        elif strip_bos and ids and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        input_ids.extend(ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
