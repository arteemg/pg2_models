"""
Conformance test suite for evedesign models.

A new model contributor inherits one or more of the contract mixins exposed
here alongside a ``model`` (and optionally ``system``) fixture to verify that
their implementation honors the framework's behavioral contracts as documented
in ``evedesign.model``.

Example
-------
.. code-block:: python

    import pytest
    from evedesign.testing import (
        BaseModelContract, ScorerContract, MutationScorerContract,
    )
    from evedesign.testing.fixtures import tiny_protein_system

    class TestMyModel(BaseModelContract, ScorerContract, MutationScorerContract):
        @pytest.fixture
        def system(self):
            return tiny_protein_system()

        @pytest.fixture
        def model(self, system):
            return MyModel(...).build(system, data=...)
"""
from evedesign.testing.contracts import (
    BaseModelContract,
    MutationScorerContract,
    ScorerContract,
)

__all__ = [
    "BaseModelContract",
    "MutationScorerContract",
    "ScorerContract",
]
