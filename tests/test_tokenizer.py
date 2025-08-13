import pytest
from little_language_model.tokenizer import tokenize

def test_tokenize_simple_sentence():
    sentence = "Hello world, this is a test."
    expected = ["Hello", "world,", "this", "is", "a", "test."]
    assert tokenize(sentence) == expected
