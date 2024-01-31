from twttr import shorten

def test_shorten():
    assert shorten("aeiou") == ""
    assert shorten("AEIOU") == ""
    assert shorten("abcde") == "bcd"
    assert shorten("twitter") == "twttr"
    assert shorten("What's your name?") == "Wht's yr nm?"
    assert shorten("tWitTer") == "tWtTr"
    assert shorten("1a2be") == "12b"