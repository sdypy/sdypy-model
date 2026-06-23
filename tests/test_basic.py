import sdypy.model


def test_namespace_import():
    """The package imports and exposes the public acoustic entry point."""
    assert hasattr(sdypy.model, "AcousticExternalProblem")
