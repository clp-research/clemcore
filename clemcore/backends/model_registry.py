import abc
import json
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Union, Tuple, Any
import importlib.resources as importlib_resources
import nltk

module_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec(SimpleNamespace):
    """Base class for model specifications.
    Holds all necessary information to make a model available for clembench: Responsible backend and any arbitrary data
    required by the backend. Also covers non-LLM 'models' like programmatic, slurk and direct user input.
    """
    PROGRAMMATIC_SPECS = ["mock", "dry_run", "programmatic", "custom", "_slurk_response"]
    HUMAN_SPECS = ["human", "terminal"]

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Keyword arguments used to set up the ModelSpec instance.
        """
        super().__init__(**kwargs)

    def unify(self, other: "ModelSpec") -> "ModelSpec":
        """Unify two ModelSpec instances.
        Args:
            other: The other ModelSpec instance this instance is to be unified with.
        Returns:
            The ModelSpec unification of this ModelSpec instance and the passed ModelSpec instance.
        Raises:
            ValueError: A ValueError exception is raised if the passed ModelSpec instance does not unify with this
                ModelSpec instance.
        """
        result = nltk.featstruct.unify(self.__dict__, other.__dict__)
        if result is None:
            raise ValueError(f"{self} does not unify with {other}")
        return ModelSpec(**result)

    def __repr__(self):
        """Get a string representation of this ModelSpec instance."""
        return f"ModelSpec({str(self)})"

    def __str__(self):
        """Get a string version of this ModelSpec instance."""
        return str(self.__dict__)

    def __getitem__(self, item):
        """Enable dict-like behavior."""
        return getattr(self, item)

    def __contains__(self, item):
        """Enable dict-like behavior."""
        return self.has_attr(item)

    def has_attr(self, attribute):
        """Check if this ModelSpec instance has the passed attribute.
        Args:
            attribute: The attribute to check for.
        """
        return hasattr(self, attribute)

    def has_temperature(self):
        """Check if this ModelSpec instance has a set 'temperature' attribute."""
        return self.has_attr("temperature")

    def has_backend(self):
        """Check if this ModelSpec instance has a set 'backend' attribute."""
        return self.has_attr("backend")

    @classmethod
    def from_name(cls, model_name: str):
        """Create a ModelSpec instance based on a model name.
        Args:
            model_name: The model name/ID as string.
        """
        if model_name is None:
            raise ValueError(f"Cannot create ModelSpec because model_name is None (but required)")
        return cls(model_name=model_name)

    @classmethod
    def from_dict(cls, spec: Dict):
        """Initialize a ModelSpec from a dictionary.
        Can be used to directly create a ModelSpec from a model registry entry dictionary.
        Args:
            spec: A model specification as dict.
        """
        return cls(**spec)

    def to_dict(self):
        return dict(self.__dict__)

    def to_string(self):
        return json.dumps(self.__dict__, separators=(",", ":"), indent=None)

    @classmethod
    def from_string(cls, model_string: str):
        """
            Get a ModelSpec instance for the passed model string.
            Takes both simple model names and (partially or fully specified) model specification data as JSON strings.
            Args:
                model_string: Model name strings correspond to the 'model_name' key value of a model in the model
                              registry. May also be partially or fully specified model specification as JSON.
            Returns:
                A ModelSpec instance
        """
        try:
            model_string = model_string.replace("'", "\"")  # make this a proper json
            model_dict = json.loads(model_string)
            return cls.from_dict(model_dict)
        except Exception as e:  # likely not a json
            return cls.from_name(model_string)

    @classmethod
    def from_strings(cls, model_strings: List[str]):
        """Get ModelSpec instances for the passed list of models.
        Takes both simple model names and (partially or fully specified) model specification data as JSON strings.
        Args:
            model_strings: List of string names of the models to return ModelSpec instances for. Model name strings
                correspond to the 'model_name' key value of a model in the model registry. May also be partially or fully
                specified model specification data as JSON strings.
        Returns:
            A list of ModelSpec instances for the passed list of models.
        """
        model_specs = []
        for model_string in model_strings:
            model_spec = cls.from_string(model_string)
            model_specs.append(model_spec)
        return model_specs

    def is_programmatic(self):
        """Check if this ModelSpec instance specifies a programmatic responder."""
        return self.model_name in ModelSpec.PROGRAMMATIC_SPECS

    def is_human(self):
        """Check if this ModelSpec instance specifies a human responder."""
        return self.model_name in ModelSpec.HUMAN_SPECS


class ModelRegistry:

    def __init__(self, model_specs: List[ModelSpec] = None):
        if model_specs is None:
            model_specs = []
        self._model_specs = model_specs

    def __len__(self):
        return len(self._model_specs)

    def __iter__(self):
        return iter(self._model_specs)

    @classmethod
    def from_packaged_and_cwd_files(cls) -> "ModelRegistry":
        """
        Lookup model_registry.json in the following locations:
        (1) Lookup in current working directory (relative to script execution)
        (2) Lookup in the packaged clemcore backends module
        Model specs found in the (1) are listed before (2) allowing to 'favor' the ones in (1).
        :return: model registry with model specs
        """
        registry = cls()
        try:
            model_registry_path = os.path.join(os.getcwd(), "model_registry.json")
            with open(model_registry_path, encoding='utf-8') as f:
                registry.register_from_list(json.load(f), lookup_source=model_registry_path)
        except Exception as e:
            module_logger.debug("File lookup failed with exception: %s", e)
        try:
            with importlib_resources.files(__package__).joinpath("model_registry.json").open("r") as f:
                registry.register_from_list(json.load(f), lookup_source="packaged")
        except Exception as e:
            module_logger.warning("Package lookup failed with exception: %s", e)
        return registry

    def register_from_list(self, model_specs: List[Dict], lookup_source: str = None) -> "ModelRegistry":
        for model_spec_dict in model_specs:
            if lookup_source:
                if "lookup_source" not in model_spec_dict:
                    model_spec_dict["lookup_source"] = lookup_source
            model_spec: ModelSpec = ModelSpec.from_dict(model_spec_dict)
            if not model_spec.has_backend():
                raise ValueError(
                    f"Missing backend definition in model spec '{model_spec}'. "
                    f"Check or update your model_registry.json and try again."
                    f"A minimal model spec is {{'model_id':<id>,'backend':<backend>}}.")
            self._model_specs.append(model_spec)
        return self

    def get_first_model_spec_that_unify_with(self, model_selector: Union[str, Dict, ModelSpec]) -> ModelSpec:
        """Get a Model subclass based on the passed specification.
        Args:
            model_selector: The model spec for which a supporting backend has to be found.
                            Can be either a model name as string,
                            a dictionary version of a model specification or a ModelSpec instance.
        Returns:
            The unified model spec that matches the model_selector.
        Raises:
            ValueError: Will be raised if the model specification does not contain fitting backend information - after
                unification with registered model specifications.
        """

        if isinstance(model_selector, str):
            model_selector = ModelSpec.from_name(model_selector)
        if isinstance(model_selector, dict):
            model_selector = ModelSpec.from_dict(model_selector)

        # for now, special handling of mock and terminal inputs (should be rather integrated via backends)
        if model_selector.is_human() or model_selector.is_programmatic():
            if model_selector.is_human():
                return ModelSpec.from_dict({"model_name": model_selector.model_name,
                                            "backend": "_player_human"})
            if model_selector.is_programmatic():
                return ModelSpec.from_dict({"model_name": model_selector.model_name,
                                            "backend": "_player_programmed"})

        if not self._model_specs:
            raise AttributeError("Model registry is empty. Load a model registry and try again.")

        selected_model_specs = []
        for registered_spec in self._model_specs:
            try:
                unified_model_spec = model_selector.unify(registered_spec)
                selected_model_specs.append(unified_model_spec)
                break  # use first model spec that does unify (doesn't throw an error)
            except ValueError:
                continue

        if not selected_model_specs:
            raise ValueError(f"No model spec unifies with model selector={model_selector.to_string()}")

        unified_model_spec = selected_model_specs[0]
        if not unified_model_spec.has_backend():
            raise ValueError(
                f"Model spec requires 'backend' after unification, but not found in model spec '{model_selector}'. "
                f"Check or update the backends/model_registry.json or pass the backend directly and try again. "
                f"A minimal model spec is {{'model_id':<id>,'backend':<backend>}}.")
        return unified_model_spec


class Model(abc.ABC):
    """A local/remote proxy for a model to be called."""

    def __init__(self, model_spec: ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance that specifies the model and the backend to be used.
        """
        assert hasattr(model_spec, "model_name"), "The passed ModelSpec must have a `model_name` attribute"
        self.model_spec = model_spec
        self.__gen_args = dict()

    def set_gen_args(self, **gen_args):
        """Set text generation inference parameters for this model.
        Currently implemented: Temperature and maximum number of tokens to generate.
        Args:
            gen_args: Keyword arguments/dict containing extra information needed for the generation process.
        """
        self.__gen_args = dict(gen_args)

    def set_gen_arg(self, arg_name, arg_value):
        """Set a text generation inference parameter for this model.
        Currently implemented: Temperature and maximum number of tokens to generate.
        Args:
            arg_name: The name of the generation inference parameter.
            arg_value: The value of the generation inference parameter.
        """
        self.__gen_args[arg_name] = arg_value

    def get_gen_arg(self, arg_name):
        """Get the value of a text generation inference parameter for this model.
        Currently implemented: Temperature and maximum number of tokens to generate.
        Args:
            arg_name: The name of the generation inference parameter.
        """
        assert arg_name in self.__gen_args, f"No '{arg_name}' in gen_args given but is expected"
        return self.__gen_args[arg_name]

    def get_temperature(self):
        """Get the value of the temperature text generation inference parameter for this model.
        Returns:
            The sampling temperature used for the generation process.
        """
        return self.get_gen_arg("temperature")

    def get_max_tokens(self):
        """Get the value of the maximum number of tokens text generation inference parameter for this model.
        Returns:
            The maximum number of tokens generated during the generation process.
        """
        return self.get_gen_arg("max_tokens")

    def get_name(self) -> str:
        """Get the name of this model.
        Returns:
            The name of the model as a string.
        """
        return self.model_spec.model_name

    def __repr__(self):
        """Get a string representation of this Model instance."""
        return str(self)

    def __str__(self):
        """Get the name of this Model instance's model.
        Returns:
            The name of the model as a string.
        """
        return self.get_name()

    def __eq__(self, other: "Model"):
        """Check if another assumed Model instance has the same model.
        Also checks if the passed object is a Model instance.
        Args:
            other: The other object to check for being a Model instance and having the same model name.
        Returns:
            False if either the passed object is not a Model instance or the passed object is a Model instance, but has
            a different model name; True if the passed object is both a Model instance and has the same model name.
        """
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


class CustomResponseModel(Model):
    """Model child class to handle custom programmatic responses."""

    def __init__(self, model_spec=ModelSpec(model_name="programmatic")):
        super().__init__(model_spec)
        self.set_gen_args(temperature=0.0)  # dummy value for get_temperature()

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player for now.")


class HumanModel(Model):
    """Model child class to handle human (terminal) responses."""

    def __init__(self, model_spec=ModelSpec(model_name="human")):
        super().__init__(model_spec)
        self.set_gen_args(temperature=0.0)  # dummy value for get_temperature()

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player for now.")
