import pytest
from little_language_model.tokenizer import data_loader

def test_data_loader():
    result = data_loader("data/the-verdict.txt")
    assert result != []
