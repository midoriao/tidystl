from tidystl import get_default_backend_registry as _get_registry

from .backend import SIMDBackend

_get_registry().register(SIMDBackend())
