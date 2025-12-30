import logging
from datetime import datetime
from typing import Dict, List
from clemcore.backends.model_registry import (
    ModelSpec,
    ModelRegistry,
    Model,
    HumanModel,
    CustomResponseModel,
    BatchGenerativeModel
)
from clemcore.backends.key_registry import KeyRegistry
from clemcore.backends.backend_registry import Backend, RemoteBackend, BackendRegistry
from clemcore.utils.log_utils import temporary_loglevel

logger = logging.getLogger(__name__)  # by default also logged to console

__all_ = [
    "Model",
    "BatchGenerativeModel",
    "ModelSpec",
    "ModelRegistry",
    "HumanModel",
    "CustomResponseModel",
    "Backend",
    "RemoteBackend",
    "BackendRegistry",
    "KeyRegistry"
]


def load_model(model_spec: str | ModelSpec, gen_args: Dict = None) -> Model:
    """
    Loads a single model which given model_spec matches one in the model registry file.

    Args:
        model_spec: either as a model_name or a ModelSpec instance
        gen_args: optional arguments to control the model's generate method

    Returns: the loaded Model as specified by the model_spec
    """
    return load_models([model_spec], gen_args)[0]


@temporary_loglevel(logger, logging.INFO)
def load_models(model_specs: List[str | ModelSpec], gen_args: Dict = None) -> List[Model]:
    """
        Loads multiple models whose given model specs each match one in the model registry file.

        Args:
            model_specs: a list of model specs, either as a model_name or a ModelSpec instance
            gen_args: optional arguments to control the model's generate method

        Returns: the list of loaded Model's as specified by the model_specs
        """
    if gen_args is None:
        gen_args = dict(temperature=0.0, max_tokens=300)
        logger.warning(f"No generation arguments provided, using default values: {gen_args}.")
    model_selectors = [
        ModelSpec.from_name(model_spec) if isinstance(model_spec, str) else model_spec
        for model_spec in model_specs
    ]
    # check models are available
    _model_registry = ModelRegistry.from_packaged_and_cwd_files()
    unified_model_specs = []
    for model_selector in model_selectors:
        unified_model_spec = _model_registry.get_first_model_spec_that_unify_with(model_selector)
        logger.info(f"Found registered model spec that unifies with {model_selector.to_string()} "
                    f"-> {unified_model_spec}")
        unified_model_specs.append(unified_model_spec)

    # check backends are available
    _backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    for unified_model_spec in unified_model_specs:
        backend_selector = unified_model_spec.backend
        if not _backend_registry.is_supported(backend_selector):
            raise ValueError(f"Specified model backend '{backend_selector}' not found in backend registry.")
        logger.info(f"Found registry entry for backend {backend_selector} "
                    f"-> {_backend_registry.get_first_file_matching(backend_selector)}")

    # ready to rumble, do the heavy lifting only now, that is, loading the additional modules
    start = datetime.now()
    player_models = []
    for unified_model_spec in unified_model_specs:
        logger.info(f"Dynamically import backend {unified_model_spec.backend}")
        backend = _backend_registry.get_backend_for(unified_model_spec.backend)
        model = backend.get_model_for(unified_model_spec)
        model.set_gen_args(**gen_args)  # todo make this somehow available in generate method?
        logger.info(f"Successfully loaded {unified_model_spec.model_name} model")
        player_models.append(model)
    logger.info("Loading models took: %s", datetime.now() - start)

    return player_models
