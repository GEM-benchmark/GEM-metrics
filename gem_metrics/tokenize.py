#!/usr/bin/env python3

import re
from functools import partial
import nltk
from .data import nltk_ensure_download


def default_tokenize_func(lang):
    """Return the default tokenizer function for a given language (Punkt, backoff to dumb_tokenize).
    @param lang: pycountry.db.Language object representing the language (result of pycountry.languages.get)
    @return tokenizer function, taking one parameter (text) and returning list of tokens.
    """
    nltk_ensure_download('tokenizers/punkt')
    func = dumb_tokenize
    if lang is not None:
        try:
            func = partial(nltk.tokenize.word_tokenize, language=lang.name.lower())
            func('.')  # this will trigger an exception if Punkt doesn't have the language
        except LookupError:
            func = dumb_tokenize  # punkt
    return func


def dumb_tokenize(text):
    """Tokenize text (separate tokens by spaces), language-agnostic failsafe version.
    @param text: String to be tokenized
    @return list of tokens
    """

    toks = text
    # separate quotes everywhere
    toks = re.sub(r'(["<>{}“”«»–|—„‚‘]|\[|\]|``|\'\'|‘‘|\^)', r' \1 ', toks)

    # the following characters (double-characters) are separated everywhere (except inside URLs)
    toks = re.sub(r'([;!()?#\$£%&*…]|--)', r' \1 ', toks)

    # short hyphen is separated if it is followed or preceeded by non-alphanuneric character and
    # is not a part of --, or a unary minus
    toks = re.sub(r'([^\-\w])\-([^\-0-9])', r'\1 - \2', toks)
    toks = re.sub(r'([0-9]\s+)\-([0-9])', r'\1 - \2', toks)  # preceded by a number - not a unary minus
    toks = re.sub(r'([^\-])\-([^\-\w])', r'\1 - \2', toks)

    # plus is separated everywhere, except at the end of a word (separated by a space) and as unary plus
    toks = re.sub(r'(\w)\+(\w)', r'\1 + \2', toks)
    toks = re.sub(r'([0-9]\s*)\+([0-9])', r'\1 + \2', toks)
    toks = re.sub(r'\+([^\w\+])', r'+ \1', toks)

    # apostrophe is separated if it is followed or preceeded by non-alphanumeric character,
    # is not part of '', and is not followed by a digit (e.g. '60).
    toks = re.sub(r'([^\'’\w])([\'’])([^\'’\d])', r'\1 \2 \3', toks)
    toks = re.sub(r'([^\'’])([\'’])([^\'’\w])', r'\1 \2 \3', toks)

    # dot, comma, slash, and colon are separated if they do not connect two numbers
    toks = re.sub(r'(\D|^)([\.,:\/])', r'\1 \2', toks)
    toks = re.sub(r'([\.,:\/])(\D|$)', r'\1 \2', toks)

    # three dots belong together
    toks = re.sub(r'\.\s*\.\s*\.', r'...', toks)

    # most common contractions
    toks = re.sub(r'([\'’´])(s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I'm, I've etc.
    toks = re.sub(r'(n[\'’´]t\s)', r' \1 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = re.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = re.sub(r' ([Dd])\'ye\s', r' \1\' ye ', toks)
    toks = re.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = re.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = re.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = re.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = re.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = re.sub(r' \'([Tt])is\s', r' \'\1 is ', toks)
    toks = re.sub(r' \'([Tt])was\s', r' \'\1 was ', toks)
    toks = re.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = re.sub(r'\s+', ' ', toks)
    toks = toks.strip()
    return toks.split(' ')

