from .language_model.grasp_llama     import GraspLlamaForCausalLM,    GraspConfig
from .language_model.grasp_mistral   import GraspMistralForCausalLM,  GraspMistralConfig


class _MissingQwen3Dependency:
    _error = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ImportError(
            "Qwen3 support requires a Transformers release that exports the "
            "Qwen3 configuration/model classes. Upgrade Transformers in the "
            "formal-training environment."
        ) from cls._error


try:
    from .language_model.grasp_qwen3 import GraspQwen3ForCausalLM, GraspQwen3Config
except ImportError as exc:  # Keep non-Qwen modules importable in older environments.
    GraspQwen3ForCausalLM = type(
        "GraspQwen3ForCausalLM", (_MissingQwen3Dependency,), {"_error": exc}
    )
    GraspQwen3Config = None

try:
    from .language_model.grasp_qwen3_moe import GraspQwen3MoEForCausalLM, GraspQwen3MoEConfig
except ImportError as exc:
    GraspQwen3MoEForCausalLM = type(
        "GraspQwen3MoEForCausalLM", (_MissingQwen3Dependency,), {"_error": exc}
    )
    GraspQwen3MoEConfig = None
