"""
Lightweight text utilities shared by the cleaning and EDA stages.

We deliberately avoid a heavy model tokenizer here: for exploratory analysis a
transparent, dependency-free word/character count is easier to reason about, and
an approximate token estimate (words x 1.3) is accurate enough to discuss context
length and compute budget without pinning us to one model's vocabulary.
"""
import html
import re
import unicodedata

# Pre-compiled patterns (compiled once, reused for every row).
_HTML_TAG = re.compile(r"<[^>]+>")           # strips stray HTML markup (MedQuAD, web text)
_WHITESPACE = re.compile(r"\s+")             # collapses runs of whitespace
_WORD = re.compile(r"\b\w[\w'-]*\b")         # a "word" for length / vocabulary counts

# Approximate ratio of sub-word tokens to whitespace words for English clinical
# text. Used only for rough context-length / compute discussion in the EDA.
TOKENS_PER_WORD = 1.3


def clean_text(text) -> str:
    """Normalise a raw string: unicode-normalise, unescape HTML entities, strip
    tags, and collapse whitespace. Returns an empty string for missing values so
    downstream length features are always numeric."""
    if text is None:
        return ""
    text = str(text)
    # NFKC folds look-alike unicode (e.g. curly quotes, ligatures) to canonical forms.
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)          # &amp; -> &, &#x27; -> ', etc.
    text = _HTML_TAG.sub(" ", text)     # remove any embedded HTML tags
    text = _WHITESPACE.sub(" ", text)   # collapse newlines/tabs/multiple spaces
    return text.strip()


def word_count(text: str) -> int:
    """Number of word tokens in a (already cleaned) string."""
    if not text:
        return 0
    return len(_WORD.findall(text))


def approx_tokens(text: str) -> int:
    """Rough sub-word token estimate for context-length / compute discussion."""
    return int(round(word_count(text) * TOKENS_PER_WORD))


def content_words(text: str, stopwords: set) -> list:
    """Lower-cased word tokens with stop-words and pure numbers removed, used to
    surface the most frequent clinical vocabulary during EDA."""
    if not text:
        return []
    out = []
    for w in _WORD.findall(text.lower()):
        if w in stopwords or w.isdigit() or len(w) < 3:
            continue
        out.append(w)
    return out
