import pytest

from xwalk._extras import MissingExtra, require


def test_require_returns_an_installed_module():
    module = require("base", "json", purpose="testing")
    assert module.dumps({"a": 1}) == '{"a": 1}'


def test_a_missing_module_raises_missing_extra():
    with pytest.raises(MissingExtra):
        require("dense", "definitely_not_installed_xyz", purpose="DenseRetriever")


def test_the_error_names_the_extra_and_the_install_command():
    with pytest.raises(MissingExtra) as exc:
        require("dense", "definitely_not_installed_xyz", purpose="DenseRetriever")
    message = str(exc.value)
    assert "xwalk[dense]" in message
    assert "pip install" in message
    assert "DenseRetriever" in message


def test_missing_extra_is_an_import_error_subclass():
    """So `except ImportError` in user code still works."""
    assert issubclass(MissingExtra, ImportError)


def test_submodules_are_importable():
    module = require("base", "os.path", purpose="testing")
    assert hasattr(module, "join")


def test_an_import_error_from_inside_a_present_module_is_not_disguised(monkeypatch):
    """A broken install is a different problem from a missing one, and saying
    'pip install xwalk[dense]' about a module that IS installed sends the user in
    circles."""
    import importlib

    def boom(name):
        raise ImportError("libcudart.so.12: cannot open shared object file", name=None)

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(MissingExtra, match="libcudart"):
        require("dense", "torch", purpose="DenseRetriever")
