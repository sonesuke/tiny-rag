import tiny_rag


def test_import():
    version = tiny_rag.__version__
    assert isinstance(version, str)
