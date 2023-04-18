""" from https://github.com/keithito/tacotron """

from text import cmudict
import params

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

chosung_list = [u"ㄱ", u"ㄲ", u"ㄴ", u"ㄷ", u"ㄸ", u"ㄹ", u"ㅁ", u"ㅂ", u"ㅃ", u"ㅅ", u"ㅆ", u"ㅇ", u"ㅈ", u"ㅉ", u"ㅊ", u"ㅋ", u"ㅌ", u"ㅍ", u"ㅎ"]
jungsung_list = [u"ㅏ", u"ㅐ", u"ㅑ", u"ㅒ", u"ㅓ", u"ㅔ", u"ㅕ", u"ㅖ", u"ㅗ", u"ㅘ", u"ㅙ", u"ㅚ", u"ㅛ", u"ㅜ", u"ㅝ", u"ㅞ", u"ㅟ", u"ㅠ", u"ㅡ", u"ㅢ", u"ㅣ"]
jongsung_list = [u"", u"ㄱ", u"ㄲ", u"ㄳ", u"ㄴ", u"ㄵ", u"ㄶ", u"ㄷ", u"ㄹ", u"ㄺ", u"ㄻ", u"ㄼ", u"ㄽ", u"ㄾ", u"ㄿ", u"ㅀ", u"ㅁ", u"ㅂ", u"ㅄ", u"ㅅ", u"ㅆ", u"ㅇ", u"ㅈ", u"ㅊ", u"ㅋ", u"ㅌ", u"ㅍ", u"ㅎ"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness:
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

if params.for_korean:
    symbols = symbols + chosung_list + jungsung_list + jongsung_list
