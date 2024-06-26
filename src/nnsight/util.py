"""Module for utility functions and classes used throughout the package."""

import importlib
import types
from functools import wraps
from typing import Any, Callable, Collection, Type

import torch

# TODO Have an Exception you can raise to stop apply early


def apply(
    data: Collection, fn: Callable, cls: Type, inplace: bool = False
) -> Collection:
    """Applies some function to all members of a collection of a give type (or types)

    Args:
        data (Collection): Collection to apply function to.
        fn (Callable): Function to apply.
        cls (type): Type or Types to apply function to.

    Returns:
        Collection: Same kind of collection as data, after then fn has been applied to members of given type.
    """
    if isinstance(data, cls):
        return fn(data)

    data_type = type(data)

    if data_type == list:
        if inplace:
            for idx, _data in enumerate(data):
                data[idx] = apply(_data, fn, cls, inplace=inplace)
            return data
        return [apply(_data, fn, cls, inplace=inplace) for _data in data]

    if data_type == tuple:
        return tuple([apply(_data, fn, cls, inplace=inplace) for _data in data])

    if data_type == dict:
        if inplace:
            for key, value in data.items():
                data[key] = apply(value, fn, cls, inplace=inplace)
            return data
        return {
            key: apply(value, fn, cls, inplace=inplace) for key, value in data.items()
        }

    if data_type == slice:
        return slice(
            apply(data.start, fn, cls, inplace=inplace),
            apply(data.stop, fn, cls, inplace=inplace),
            apply(data.step, fn, cls, inplace=inplace),
        )

    return data


def fetch_attr(object: object, target: str) -> Any:
    """Retrieves an attribute from an object hierarchy given an attribute path. Levels are separated by '.' e.x (transformer.h.1)

    Args:
        object (object): Root object to get attribute from.
        target (str): Attribute path as '.' separated string.

    Returns:
        Any: Fetched attribute.
    """
    if target == "":
        return object

    target_atoms = target.split(".")

    for atom in target_atoms:

        if not atom:
            continue

        object = getattr(object, atom)

    return object


def wrap(object: object, wrapper: Type, *args, **kwargs) -> object:
    """Wraps some object given some wrapper type.
    Updates the __class__ attribute of the object and calls the wrapper type's __init__ method.

    Args:
        object (object): Object to wrap.
        wrapper (Type): Type to wrap the object in.

    Returns:
        object: Wrapped object.
    """
    if isinstance(object, wrapper):
        return object

    new_class = types.new_class(
        object.__class__.__name__,
        (object.__class__, wrapper),
    )

    object.__class__ = new_class

    wrapper.__init__(object, *args, **kwargs)

    return object


def to_import_path(type: type) -> str:

    return f"{type.__module__}.{type.__name__}"


def from_import_path(import_path: str) -> type:

    *import_path, classname = import_path.split(".")
    import_path = ".".join(import_path)

    return getattr(importlib.import_module(import_path), classname)


class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args
    
def wrap_object_as_module(obj):
    class_name = f"Wrapped{obj.__class__.__name__}"

    class WrappedModule(torch.nn.Module):
        def __init__(self, wrapped_obj):
            super().__init__()
            self.obj = wrapped_obj
            
            for name in dir(self.obj):
                if not name.startswith("_"):
                    method = getattr(self.obj, name)
                    if callable(method):
                        wrapped_method = self._wrap_method(method)
                        setattr(self, name, wrapped_method)

        def _wrap_method(self, method):
            method_module = torch.nn.Module()
            def forward(*args, **kwargs):
                return method(*args, **kwargs)
            method_module.forward = forward
            return method_module

        def forward(self, method_name, *args, **kwargs):
            method = getattr(self, method_name, None)
            if isinstance(method, torch.nn.Module):
                return method(*args, **kwargs)
            if callable(method):
                return method(*args, **kwargs)
            raise AttributeError(f"'{class_name}' object has no attribute '{method_name}'")

    WrappedModule.__name__ = WrappedModule.__qualname__ = class_name
    return WrappedModule(obj)