import abc
import importlib
import inspect
import os
import importlib.resources as importlib_resources
import importlib.util as importlib_util
from typing import List

from clemcore.backends import ModelSpec, Model, HumanModel, CustomResponseModel
from clemcore.backends.key_registry import KeyRegistry


class Backend(abc.ABC):
    """
    A backend is a factory for creating models of a particular model provider.

    Classes that implement a Backend are automatically detected by the BackendRegistry.
    """

    @abc.abstractmethod
    def get_model_for(self, model_spec: ModelSpec) -> Model:
        """Get a Model instance for the model specific by ModelSpec.
        Must be implemented by every clembench backend.
        Args:
            model_spec: A ModelSpec instance specifying the model to return a corresponding Model child class instance
                for the appropriate backend.
        Returns:
            A Model instance using the appropriate backend.
        """
        pass

    def __repr__(self):
        """Get a string representation of this Backend instance."""
        return str(self)

    def __str__(self):
        """Get a string name of the class of this Backend child class instance."""
        return f"{self.__class__.__name__}"


class RemoteBackend(Backend):

    def __init__(self, key_name: str = None):
        self.key_name = key_name or self.__class__.__name__.lower()
        self.key = KeyRegistry.from_json().get_key_for(self.key_name)
        self.client = self._make_api_client()

    @abc.abstractmethod
    def _make_api_client(self):
        """Subclasses must return an initialized client for remote interaction."""
        pass


def is_backend(obj):
    """Check if an object is a Backend child class (instance).
    Args:
        obj: The object to be checked.
    Returns:
        True if the object is a Backend child class (instance); False otherwise.
    """
    if inspect.isclass(obj) and issubclass(obj, Backend) and obj is not Backend:
        return True
    return False


def to_backend_name(file_name: str):
    return file_name.replace("_api.py", "")


def is_backend_file(file_name: str):
    return file_name.endswith("_api.py")


class BackendRegistry:

    def __init__(self, backend_files: List):
        self._backends_files = backend_files
        # for now, special handling of mock and terminal inputs (should be rather integrated via backends)
        self._backends_files.append({
            "backend": "_player_human",
            "file_name": "_internal",
            "file_path": "_internal",
            "lookup_source": "packaged"
        })
        self._backends_files.append({
            "backend": "_player_programmed",
            "file_name": "_internal",
            "file_path": "_internal",
            "lookup_source": "packaged"
        })

    def __len__(self):
        return len(self._backends_files)

    def __iter__(self):
        return iter(self._backends_files)

    def is_supported(self, backend_name: str):
        for backend_file in self._backends_files:
            if backend_file["backend"] == backend_name:
                return True
        return False

    def get_first_file_matching(self, backend_selector: str):
        for backend_file in self._backends_files:
            if backend_file["backend"] == backend_selector:
                return backend_file
        raise ValueError(f"No registered backend file found for selector={backend_selector}")

    @classmethod
    def from_packaged_and_cwd_files(cls) -> "BackendRegistry":
        """
        Lookup _api.py files in the following locations:
        (1) Lookup in current working directory (relative to script execution)
        (2) Lookup in the packaged clemcore backends module
        Backends found in the (1) are favored over (2), that is listed before them to allow to 'overwrite' them.
        :return: backend registry with file path to backends to be dynamically loaded
        """
        backend_files = []
        for file in os.listdir():
            if is_backend_file(file):
                backend_files.append({"backend": to_backend_name(file),
                                      "file_name": file,
                                      "file_path": os.path.join(os.getcwd(), file),
                                      "lookup_source": "cwd"})
        for file in importlib_resources.files(__package__).iterdir():  # __package__ already points to "backends"
            if is_backend_file(file.name):
                backend_files.append({"backend": to_backend_name(file.name),
                                      "file_name": file.name,
                                      "file_path": str(file),
                                      "lookup_source": "packaged"})

        return cls(backend_files)

    def get_backend_for(self, backend_selector: str) -> Backend:
        """Dynamically loads the Backend from the first file that matches the name <backend_selector>_api.py.
        Raises an exception if no such file exists or the Backend class could not be found.
        Args:
            backend_selector: The <backend_selector> prefix of the <backend_selector>_api.py file.
        Returns:
            The Backend subclass for the passed backend name.
        Raises:
            FileNotFoundError: Will be raised if no backend python file with the passed name can be found in the backends
                directory.
            LookupError: Will be raised if the backend python file with the passed name does not contain exactly one Backend
                subclass.
        """
        # for now, special handling of mock and terminal inputs (should be rather integrated via backends)
        if backend_selector == "_player_human":
            return HumanModelBackend()
        if backend_selector == "_player_programmed":
            return CustomResponseModelBackend()

        backend_file = self.get_first_file_matching(backend_selector)
        module_name, _ = os.path.splitext(backend_file["file_name"])
        if backend_file["lookup_source"] == "packaged":
            # relative imports (sibling to __init__.py) require starting dot
            module = importlib.import_module(f".{module_name}", __package__)
        else:
            module_path = backend_file["file_path"]
            spec = importlib_util.spec_from_file_location(module_name, module_path)
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
        backend_subclasses = inspect.getmembers(module, predicate=is_backend)
        if len(backend_subclasses) == 0:
            raise LookupError(f"There is no Backend defined in {module}. "
                              f"Create such a class and try again or "
                              f"check the backend selector '{backend_selector}'.")
        if len(backend_subclasses) > 1:
            raise LookupError(f"There is more than one Backend defined in {module}.")
        _, backend_cls = backend_subclasses[0]
        return backend_cls()


class HumanModelBackend(Backend):

    def get_model_for(self, model_spec: ModelSpec) -> Model:
        if model_spec.is_human():
            return HumanModel(model_spec)
        raise ValueError(f"HumanModelBackend cannot get model for {model_spec.to_string()}")


class CustomResponseModelBackend(Backend):

    def get_model_for(self, model_spec: ModelSpec) -> Model:
        if model_spec.is_programmatic():
            return CustomResponseModel(model_spec)
        raise ValueError(f"CustomResponseModelBackend cannot get model for {model_spec.to_string()}")
