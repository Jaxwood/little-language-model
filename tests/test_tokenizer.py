import pytest
from little_language_model.tokenizer import data_loader

def test_data_loader():
    sut = data_loader("data/the-verdict.txt")
    it = iter(sut)
    assert next(it) is not None
