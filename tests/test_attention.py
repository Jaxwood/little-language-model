import pytest
from little_language_model.attention import Attention
from little_language_model.tokenizer import data_loader

def test_attention():
    data = data_loader("data/the-verdict.txt")
    it = iter(data)
    inputs, _ = next(it)
    sut = Attention(inputs.shape[1], 4)
    output = sut(inputs)
    assert output is not None
    assert output.shape[0] == inputs.shape[0]
    assert output.shape[1] == 4

