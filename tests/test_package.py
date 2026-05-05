from lih_repro import __version__


def test_package_version_is_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
