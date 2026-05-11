from kiwipiepy import Kiwi

_kiwi = Kiwi()


def korean_tokenize(text: str) -> list[str]:
    """Tokenize Korean text, keeping nouns, verbs, foreign words, and numbers."""
    tokens = []
    for t in _kiwi.tokenize(text):
        if t.tag.startswith(("N", "V", "SL", "SN")):
            tokens.append(t.form.lower())
    return tokens
