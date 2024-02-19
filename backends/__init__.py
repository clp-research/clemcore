import abc
import importlib
import inspect
import json
import os
import nltk
import logging
import logging.config
from types import SimpleNamespace
from dataclasses import dataclass

from typing import Dict, List, Tuple, Any, Type, Union

import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
with open(os.path.join(project_root, "logging.yaml")) as f:
    conf = yaml.safe_load(f)
    log_fn = conf["handlers"]["file_handler"]["filename"]
    log_fn = os.path.join(project_root, log_fn)
    conf["handlers"]["file_handler"]["filename"] = log_fn
    logging.config.dictConfig(conf)


def get_logger(name):
    return logging.getLogger(name)


# Load backend dynamically from "backends" sibling directory
# Note: The backends might use get_logger (circular import)
def load_credentials(backend, file_name="key.json") -> Dict:
    key_file = os.path.join(project_root, file_name)
    with open(key_file) as f:
        creds = json.load(f)
    assert backend in creds, f"No '{backend}' in {file_name}. See README."
    assert "api_key" in creds[backend], f"No 'api_key' in {file_name}. See README."
    return creds


@dataclass(frozen=True)
class ModelSpec(SimpleNamespace):
    PROGRAMMATIC_SPECS = ["mock", "dry_run", "programmatic", "custom", "_slurk_response"]
    HUMAN_SPECS = ["human", "terminal"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def unify(self, other: "ModelSpec") -> "ModelSpec":
        """ Return whether the other ModelSpec is fully contained within this ModelSpec """
        result = nltk.featstruct.unify(self.__dict__, other.__dict__)
        if result is None:
            raise ValueError(f"{self} does not unify with {other}")
        return ModelSpec(**result)

    def __repr__(self):
        return f"ModelSpec({str(self)})"

    def __str__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        """ dict-like behavior """
        return getattr(self, item)

    def __contains__(self, item):
        """ dict-like behavior """
        return self.has_attr(item)

    def has_attr(self, attribute):
        return hasattr(self, attribute)

    def has_temperature(self):
        return self.has_attr("temperature")

    def has_backend(self):
        return self.has_attr("backend")

    @classmethod
    def from_name(cls, model_name: str):
        if model_name is None:
            raise ValueError(f"Cannot create ModelSpec because model_name is None (but required)")
        return cls(model_name=model_name)

    @classmethod
    def from_dict(cls, spec: Dict):
        return cls(**spec)

    def is_programmatic(self):
        return self.model_name in ModelSpec.PROGRAMMATIC_SPECS

    def is_human(self):
        return self.model_name in ModelSpec.HUMAN_SPECS


class Model(abc.ABC):
    """ A local/remote proxy for a model to be called. """

    def __init__(self, model_spec: ModelSpec):
        """

        :param model_spec: that specifies the model and the backend to be used
        """
        assert hasattr(model_spec, "model_name"), "The passed ModelSpec must have a `model_name` attribute"
        self.model_spec = model_spec
        self.__gen_args = dict()

    def set_gen_args(self, **gen_args):
        """
        :param gen_args: set extra information needed for the generation process
        """
        self.__gen_args = dict(gen_args)

    def set_gen_arg(self, arg_name, arg_value):
        """ Set a particular argument needed for the generation process
        :param arg_name: the name of the generation argument
        :param arg_value: the value of the generation argument
        """
        self.__gen_args[arg_name] = arg_value

    def get_gen_arg(self, arg_name):
        assert arg_name in self.__gen_args, f"No '{arg_name}' in gen_args given but is expected"
        return self.__gen_args[arg_name]

    def get_temperature(self):
        """
        :return: the sampling temperature used for the generation process
        """
        return self.get_gen_arg("temperature")

    def get_max_tokens(self):
        """
        :return: the maximal number of tokens generated during the generation process
        """
        return self.get_gen_arg("max_tokens")

    def get_name(self) -> str:
        return self.model_spec.model_name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.get_name()

    def __eq__(self, other: "Model"):
        if not isinstance(other, Model):
            return False
        return self.get_name() == other.get_name()

    @abc.abstractmethod
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Put prompt in model-specific format and get its response.

        Args:
            messages (List[Dict]): The dialogue context represented as a list
                of turns. Entry element is a dictionary containing one key
                "role", whose value is either "user" or "assistant", and one
                key "content", whose value is the message as a string.

        Returns:
            Tuple[Any, Any, str]: The first element is the prompt object as
            passed to the LLM (i.e. after any model-specific manipulation).
            Return the full prompt object, not only the message string.

            The second element is the response object as gotten from the model,
            before any manipulation. Return the full prompt object, not only
            the message string.

            These must be returned just to be logged by the GM for later inspection.

            The third element is the response text, i.e. only the actual message
            generated by the model as a string (after any needed manipulation,
            like .strip() or excluding the input prompt).
        """
        pass


class Backend(abc.ABC):
    """ Marker class for a model provider."""

    @abc.abstractmethod
    def get_model_for(self, model_spec: ModelSpec) -> Model:
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.__class__.__name__}"


class CustomResponseModel(Model):

    def __init__(self, model_spec=ModelSpec(model_name="programmatic")):
        super().__init__(model_spec)
        self.set_gen_args(temperature=0.0)  # dummy value for get_temperature()

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player for now.")


class HumanModel(Model):

    def __init__(self, model_spec=ModelSpec(model_name="human")):
        super().__init__(model_spec)
        self.set_gen_args(temperature=0.0)  # dummy value for get_temperature()

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player for now.")


def is_backend(obj):
    if inspect.isclass(obj) and issubclass(obj, Backend):
        return True
    return False


_backend_registry: Dict[str, Backend] = dict()  # we store references to the class constructor
_model_registry: List[ModelSpec] = list()  # we store model specs so that users might use model_name for lookup


def load_custom_model_registry(_model_registry_path: str = None, is_optional=True):
    if not _model_registry_path:
        _model_registry_path = os.path.join(project_root, "backends", "model_registry_custom.json")
    load_model_registry(_model_registry_path, is_mandatory=not is_optional)


def load_model_registry(_model_registry_path: str = None, is_mandatory=True):
    if not _model_registry_path:
        _model_registry_path = os.path.join(project_root, "backends", "model_registry.json")
    if not os.path.isfile(_model_registry_path):
        if is_mandatory:
            raise FileNotFoundError(f"The file model registry at '{_model_registry_path}' does not exist. "
                                    f"Create model registry as a model_registry.json file and try again.")
        else:
            return  # do nothing
    with open(_model_registry_path) as f:
        _model_listing = json.load(f)
        for _model_entry in _model_listing:
            _model_spec: ModelSpec = ModelSpec.from_dict(_model_entry)
            if not _model_spec.has_backend():
                raise ValueError(
                    f"Missing backend definition in model spec '{_model_spec}'. "
                    f"Check or update the backends/model_registry.json and try again."
                    f"A minimal model spec is {{'model_id':<id>,'backend':<backend>}}.")
            _model_registry.append(_model_spec)


def _register_backend(backend_name: str):
    """
    Dynamically loads the Backend in the file with name <backend_name>_api.py into the _backend_registry.
    Raises an exception if no such file exists or the Backend class could not be found.

    :param backen_name: the prefix of the <backend_name>_api.py file
    """
    backends_root = os.path.join(project_root, "backends")
    backend_module = f"{backend_name}_api"
    backend_path = os.path.join(backends_root, f"{backend_module}.py")
    if not os.path.isfile(backend_path):
        raise FileNotFoundError(f"The file '{backend_path}' does not exist. "
                                f"Create such a backend file or check the backend_name '{backend_name}'.")
    module = importlib.import_module(f"backends.{backend_module}")
    backend_subclasses = inspect.getmembers(module, predicate=is_backend)
    if len(backend_subclasses) == 0:
        raise LookupError(f"There is no Backend defined in {backend_module}. "
                          f"Create such a class and try again or check the backend_name '{backend_name}'.")
    if len(backend_subclasses) > 1:
        raise LookupError(f"There is more than one Backend defined in {backend_module}.")
    _, backend_cls = backend_subclasses[0]
    _backend_registry[backend_name] = backend_cls()
    return backend_cls


def _load_model_for(model_spec: ModelSpec) -> Model:
    backend_name = model_spec.backend
    if backend_name not in _backend_registry:
        _register_backend(backend_name)
    backend_cls = _backend_registry[backend_name]
    return backend_cls.get_model_for(model_spec)


def get_model_for(model_spec: Union[str, Dict, ModelSpec]) -> Model:
    """
    :param model_spec: the model spec for which a supporting backend has to be found
    :return: the backend registered that supports the model
    """
    assert len(_model_registry) > 0, "Model registry is empty. Load a model registry and try again."

    if isinstance(model_spec, str):
        model_spec = ModelSpec.from_name(model_spec)
    if isinstance(model_spec, dict):
        model_spec = ModelSpec.from_dict(model_spec)

    if model_spec.is_human():
        return HumanModel(model_spec)
    if model_spec.is_programmatic():
        return CustomResponseModel(model_spec)

    for registered_spec in _model_registry:
        try:
            model_spec = model_spec.unify(registered_spec)
            break  # use first model spec that does unify (doesn't throw an error)
        except ValueError:
            continue

    if not model_spec.has_backend():
        raise ValueError(
            f"Model spec requires 'backend' after unification, but not found in model spec '{model_spec}'. "
            f"Check or update the backends/model_registry.json or pass the backend directly and try again. "
            f"A minimal model spec is {{'model_id':<id>,'backend':<backend>}}.")
    model = _load_model_for(model_spec)
    return model


class ContextExceededError(Exception):
    """
    Exception to be raised when the messages passed to a backend instance exceed the context limit of the model.
    """
    tokens_used: int = int()
    tokens_left: int = int()
    context_size: int = int()

    def __init__(self, info_str: str = "Context limit exceeded", tokens_used: int = 0,
                 tokens_left: int = 0, context_size: int = 0):
        info = f"{info_str} {tokens_used}/{context_size}"
        super().__init__(info)
        self.tokens_used = tokens_used
        self.tokens_left = tokens_left
        self.context_size = context_size
