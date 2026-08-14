from __future__ import annotations

import torch


@torch.inference_mode()
def binary_answer_probability(model, tokenizer, prompt_ids, *, model_kwargs=None) -> float:
    """Length-normalized likelihood for the canonical Fraud/Normal sequences."""
    model_kwargs = dict(model_kwargs or {})
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    scores = []
    for answer in ("Fraud", "Normal"):
        answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = full.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        output = model(input_ids=full, labels=labels, **model_kwargs)
        # CrossEntropyLoss is already averaged over answer tokens. Keeping the
        # mean makes this a length-normalized sequence score.
        scores.append(-output.loss.float())
    return float(torch.softmax(torch.stack(scores), dim=0)[0].cpu())
