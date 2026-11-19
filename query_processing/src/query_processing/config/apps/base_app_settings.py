"""The base of an App's flat settings: a field's variable is the prefix and its name.

The settings one App owns (``QP_OVMS_*``, ``QP_VLLM_*``, ``QP_UVICORN_*``,
``QP_PROCESS_*``) are flat variables, each the class's ``env_prefix`` and the
upper-cased field, so a message naming one is written from this one rule.

Example:
    >>> UvicornSettings.variable_of("backlog")
    'QP_UVICORN_BACKLOG'
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class BaseAppSettings(BaseSettings):
    """Flat settings of one App, read under the subclass's ``env_prefix``.

    Example:
        >>> OvmsSettings.variable_of("threads")
        'QP_OVMS_THREADS'
    """

    @classmethod
    def variable_of(cls, field: str) -> str:
        """Return the environment variable a field is read from.

        Args:
            field: The field name.

        Returns:
            The class's ``env_prefix`` followed by the upper-cased field.

        Raises:
            ValueError: If ``field`` is not a field of the class; it names the
                class, the name and the fields there are.

        Example:
            >>> ProcessEnvironmentSettings.variable_of("omp_proc_bind")
            'QP_PROCESS_OMP_PROC_BIND'
        """
        if field not in cls.model_fields:
            raise ValueError(
                f"{cls.__name__} has no field {field!r}; expected one of: "
                f"{', '.join(map(repr, cls.model_fields))}"
            )
        return cls.model_config.get("env_prefix", "") + field.upper()
