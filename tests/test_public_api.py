import konen


def test_package_imports() -> None:
    assert konen.__version__ == "0.1.0"


def test_public_exports() -> None:
    assert konen.__all__
    for name in konen.__all__:
        assert getattr(konen, name) is not None
