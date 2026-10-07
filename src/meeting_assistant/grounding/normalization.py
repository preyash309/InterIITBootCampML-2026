"""Retrieval-only normalization. Original text is never edited."""

import re
import unicodedata

PROTECTED = frozenset(
    "not never must should will won't cannot can't don't doesn't didn't isn't aren't wasn't weren't couldn't shouldn't wouldn't mustn't approved rejected agreed deadline percent percentage tomorrow yesterday today".split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    # These symbols distinguish real names: C, C++, C#, U-Net and UNet++.
    text = (
        text.replace("+", " plus ")
        .replace("#", " sharp ")
        .replace("*", " star ")
        .replace("&", " and ")
    )
    # Collapse dotted/spaced acronym letters without merging ordinary words.
    text = re.sub(
        r"\b(?:[a-z][.\s]){2,}[a-z]\.?(?=\W|$)", lambda m: re.sub(r"[^a-z]", "", m[0]), text
    )
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def protected(text: str) -> bool:
    tokens = re.findall(r"[\w']+", text.casefold().replace("’", "'"))
    return any(token in PROTECTED for token in tokens)


def phonetic(text: str) -> str:
    import jellyfish

    return jellyfish.metaphone(normalize(text).replace(" ", ""))
