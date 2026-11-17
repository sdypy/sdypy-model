"""Deprecated aliases for renamed public names (SEP 2).

The old names keep working through v1.x and are removed no earlier than v2.0.
Each renamed argument is handled inline in its function (an ``if`` on the old
name that calls ``warnings.warn(..., DeprecationWarning)``), which is the form
the hub's nomenclature checker recognises. This module holds the parts that
are the same everywhere.
"""
import warnings


class DeprecatedAttribute:
    """Class-level alias that forwards an old attribute name to its new name.

    Reading, writing or deleting the old name emits ``DeprecationWarning`` and
    acts on the new attribute, so both names always hold the same value.

    Parameters
    ----------
    new_name : str
        Name of the attribute that holds the value.

    Examples
    --------
    >>> class Beam:
    ...     nat_freq = DeprecatedAttribute("natural_freq")
    """

    def __init__(self, new_name):
        self.new_name = new_name

    def __set_name__(self, owner, name):
        self.old_name = name

    def _warn(self):
        warnings.warn(
            f"{self.old_name} is deprecated; use {self.new_name} instead",
            DeprecationWarning,
            stacklevel=3,
        )

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self._warn()
        return getattr(obj, self.new_name)

    def __set__(self, obj, value):
        self._warn()
        setattr(obj, self.new_name, value)

    def __delete__(self, obj):
        self._warn()
        delattr(obj, self.new_name)


def reject_both(old_name, new_name, new_value):
    """Raise when a renamed argument is passed under both names.

    Called after the deprecated name was found to be passed.

    Parameters
    ----------
    old_name, new_name : str
        Deprecated and canonical argument names, used in the message.
    new_value : object
        Value of the canonical argument; ``None`` means it was not passed.

    Raises
    ------
    TypeError
        When the canonical argument was passed as well.
    """
    if new_value is not None:
        raise TypeError(
            f"Cannot pass both '{new_name}' and deprecated '{old_name}'. "
            f"Use '{new_name}' only."
        )


def require(function_name, **arguments):
    """Raise when a required argument is still ``None``.

    Arguments that were renamed default to ``None`` so that the old keyword can
    stand in for them; this reports the ones that were passed under neither
    name.

    Parameters
    ----------
    function_name : str
        Name used in the message.
    **arguments
        Canonical argument names and their values after aliases are resolved.

    Raises
    ------
    TypeError
        When one or more values are ``None``.
    """
    missing = [name for name, value in arguments.items() if value is None]
    if missing:
        raise TypeError(
            f"{function_name}() missing required argument(s): "
            + ", ".join(f"'{name}'" for name in missing)
        )
