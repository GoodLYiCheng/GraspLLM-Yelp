from __future__ import annotations

import torch


@torch.inference_mode()
def binary_answer_probability(model, tokenizer, prompt_ids, *, model_kwargs=None) -> float:
    """Score canonical answer sequences instead of parsing generated prose."""
    model_kwargs = dict(model_kwargs or {})
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    def sequence_score(answer: str) -> torch.Tensor:
        answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = full.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        # Using labels lets GraspLLM insert graph embeddings and matching ignore
        # labels before the causal-LM shift. Raw token positions are therefore
        # never used to locate the answer after multimodal expansion.
        output = model(input_ids=full, labels=labels, **model_kwargs)
        return -output.loss.reshape(1) * answer_ids.shape[1]

    fraud = sequence_score("Fraudulent")
    legitimate = sequence_score("Legitimate")
    return float(torch.softmax(torch.stack([fraud, legitimate], dim=-1), dim=-1)[0, 0])
