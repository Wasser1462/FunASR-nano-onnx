"""
Hotword correction module for ASR post-processing.

This module provides phoneme-based hotword correction and history-based rectification
for automatic speech recognition (ASR) outputs. It supports both Chinese (pinyin-based)
and English (token-based) hotword matching with fuzzy matching capabilities.

Features:
- Phoneme-based similarity matching for Chinese hotwords
- Token-based pattern matching for English hotwords
- History-based correction using past correction records
- Configurable thresholds and scoring mechanisms

This implementation is based on and optimized from the hotword correction module
in CapsWriter-Offline: https://github.com/HaujetZhao/CapsWriter-Offline

Adapted and enhanced for FunASR-nano-onnx inference pipeline.
"""

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

try:
    from pypinyin import pinyin, Style  # type: ignore
    HAS_PYPINYIN = True
except Exception:
    pinyin = None
    Style = None
    HAS_PYPINYIN = False

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

try:
    from numba import njit  # type: ignore
    HAS_NUMBA = True
except Exception:
    njit = None
    HAS_NUMBA = False


SIMILAR_PHONEMES = [
    {"an", "ang"}, {"en", "eng"}, {"in", "ing"}, {"ian", "iang"}, {"uan", "uang"},
    {"z", "zh"}, {"c", "ch"}, {"s", "sh"},
    {"l", "n"}, {"f", "h"},
    {"ai", "ei"}, {"o", "uo"}, {"e", "ie"},
    {"p", "t"}, {"p", "b"}, {"t", "d"}, {"k", "g"},
]

_RE_ALNUM = re.compile(r"^[a-z0-9]+$", re.IGNORECASE)
_PINYIN_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]


@dataclass(frozen=True, slots=True)
class Phoneme:
    value: str
    lang: Literal["zh", "en", "num", "other"]
    is_word_start: bool = False
    is_word_end: bool = False
    char_start: int = 0
    char_end: int = 0

    @property
    def is_tone(self) -> bool:
        return self.value.isdigit()

    @property
    def info(self) -> Tuple[str, str, bool, bool, bool, int, int]:
        return (self.value, self.lang, self.is_word_start, self.is_word_end, self.is_tone, self.char_start, self.char_end)


@dataclass(frozen=True, slots=True)
class HotwordEntry:
    word: str
    py_hint: Optional[str] = None
    alias: Tuple[str, ...] = ()
    prefix: Tuple[str, ...] = ()
    suffix: Tuple[str, ...] = ()
    allow_eos: bool = False
    min_text_sim: Optional[float] = None
    min_span_len: int = 1


class CorrectionResult(NamedTuple):
    text: str
    matches: List[Tuple[str, str, float]]
    similars: List[Tuple[str, str, float]]


class MatchResult(NamedTuple):
    start: int
    end: int
    score: float
    hotword: str


def _is_cjk(c: str) -> bool:
    return "\u4e00" <= c <= "\u9fff"


def _is_punct_or_space(c: str) -> bool:
    return c.isspace() or c in ",.;:!?，。；：！？、()（）[]【】{}\"'“”‘’<>《》"


def _is_ascii_word(s: str) -> bool:
    if not s:
        return False
    has_alpha = False
    for ch in s:
        if ord(ch) >= 128:
            return False
        if ch.isalpha():
            has_alpha = True
    return has_alpha

def _tokenize_with_offsets(text: str) -> List[Tuple[str, str, int, int]]:
    out: List[Tuple[str, str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if _is_cjk(c):
            out.append((c, "zh", i, i + 1))
            i += 1
            continue
        if c.isalpha() or c.isdigit():
            st = i
            i += 1
            while i < n:
                cc = text[i]
                if not (cc.isalpha() or cc.isdigit()):
                    break
                if (text[i - 1].islower() and cc.isupper()) or (text[i - 1].isalpha() and cc.isdigit()) or (text[i - 1].isdigit() and cc.isalpha()):
                    break
                i += 1
            tk = text[st:i].lower()
            lang = "num" if tk.isdigit() else "en"
            out.append((tk, lang, st, i))
            continue
        i += 1
    return out


def _en_tokens(span: str) -> List[str]:
    return [t for (t, lang, _, _) in _tokenize_with_offsets(span) if lang in ("en", "num")]


def _span_has_cjk(span: str) -> bool:
    for ch in span:
        if _is_cjk(ch):
            return True
    return False


def _text_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _lcs_length(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    if n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        a = s1[i - 1]
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a == s2[j - 1] else (prev[j] if prev[j] >= curr[j - 1] else curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def _get_tuple_cost(t1: Tuple, t2: Tuple) -> float:
    if t1[1] != t2[1]:
        return 1.0
    if t1[0] == t2[0]:
        return 0.0

    lang = t1[1]
    if lang == "zh":
        if t1[4] and t2[4]:
            return 0.25
        if t1[4] != t2[4]:
            return 1.0
        pair = {t1[0], t2[0]}
        for s in SIMILAR_PHONEMES:
            if pair.issubset(s):
                return 0.5
        return 1.0

    if lang in ("en", "num"):
        lcs = _lcs_length(t1[0], t2[0])
        den = max(len(t1[0]), len(t2[0]))
        return 1.0 - (lcs / den) if den else 1.0

    return 1.0


def fuzzy_substring_search_constrained(hw_info: List[Tuple], input_info: List[Tuple], threshold: float = 0.6) -> List[Tuple[float, int, int]]:
    n, m = len(hw_info), len(input_info)
    if n == 0 or m == 0:
        return []

    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    startj = [[0] * (m + 1) for _ in range(n + 1)]

    for j in range(m + 1):
        if j == 0 or (j < m and input_info[j][2]):
            dp[0][j] = 0.0
            startj[0][j] = j

    for i in range(1, n + 1):
        hi = hw_info[i - 1]
        for j in range(1, m + 1):
            cost = _get_tuple_cost(hi, input_info[j - 1])
            a = dp[i - 1][j - 1] + cost
            b = dp[i - 1][j] + 1.0
            c = dp[i][j - 1] + 1.0

            best = a
            sj = startj[i - 1][j - 1]
            if b < best:
                best = b
                sj = startj[i - 1][j]
            if c < best:
                best = c
                sj = startj[i][j - 1]

            dp[i][j] = best
            startj[i][j] = sj

    results: List[Tuple[float, int, int]] = []
    for j in range(1, m + 1):
        if not input_info[j - 1][3]:
            continue
        dist = dp[n][j]
        if dist >= n * 0.85:
            continue
        score = 1.0 - (dist / n)
        if score >= threshold:
            results.append((score, startj[n][j], j))

    results.sort(key=lambda x: x[0], reverse=True)
    best_by_end: Dict[int, Tuple[float, int, int]] = {}
    for s, a, b in results:
        if b not in best_by_end or s > best_by_end[b][0]:
            best_by_end[b] = (s, a, b)
    return sorted(best_by_end.values(), key=lambda x: x[0], reverse=True)


def _parse_hotword_line(line: str) -> Optional[HotwordEntry]:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if not parts:
        return None
    word = parts[0]
    py_hint: Optional[str] = None
    alias: List[str] = []
    prefix: List[str] = []
    suffix: List[str] = []
    allow_eos = False
    min_text_sim: Optional[float] = None
    min_span_len = 1

    if len(parts) == 2 and ("=" not in parts[1]):
        hint = parts[1]
        if any(ch.isdigit() for ch in hint) or (" " in hint):
            py_hint = hint
        else:
            alias = [hint]
        return HotwordEntry(word, py_hint, tuple(alias), tuple(prefix), tuple(suffix), allow_eos, min_text_sim, min_span_len)

    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k in ("py", "pinyin"):
            py_hint = v
        elif k == "alias":
            alias = [x.strip() for x in v.split(",") if x.strip()]
        elif k == "prefix":
            prefix = [x.strip() for x in v.split(",") if x.strip()]
        elif k == "suffix":
            suffix = [x.strip() for x in v.split(",") if x.strip()]
        elif k == "allow_eos":
            allow_eos = v.lower() in ("1", "true", "yes", "y")
        elif k in ("min_text_sim", "text_sim_min"):
            try:
                min_text_sim = float(v)
            except Exception:
                min_text_sim = None
        elif k in ("min_span_len", "span_len_min"):
            try:
                min_span_len = int(v)
            except Exception:
                min_span_len = 1

    return HotwordEntry(word, py_hint, tuple(alias), tuple(prefix), tuple(suffix), allow_eos, min_text_sim, min_span_len)


def _phonemes_from_py_hint(hint: str) -> List[Phoneme]:
    parts = [x.strip() for x in re.split(r"[\s,]+", hint.strip()) if x.strip()]
    out: List[Phoneme] = []
    for syl in parts:
        s = syl.lower()
        tone = "5"
        if s and s[-1].isdigit():
            tone = s[-1]
            s = s[:-1]
        init = ""
        fin = s
        for cand in _PINYIN_INITIALS:
            if s.startswith(cand) and len(cand) > len(init):
                init = cand
        if init:
            fin = s[len(init):]
            if init:
                out.append(Phoneme(init, "zh", True, False))
            if fin:
                out.append(Phoneme(fin, "zh", False, False))
        else:
            if fin:
                out.append(Phoneme(fin, "zh", True, False))
        out.append(Phoneme(tone, "zh", False, True))
    return out


def get_phoneme_info(text: str) -> List[Phoneme]:
    toks = _tokenize_with_offsets(text)
    if not toks:
        return []
    seq: List[Phoneme] = []
    if not HAS_PYPINYIN:
        for v, lang, st, ed in toks:
            seq.append(Phoneme(v, lang if lang != "zh" else "zh", True, True, st, ed))
        return seq

    for v, lang, st, ed in toks:
        if lang != "zh":
            seq.append(Phoneme(v, lang, True, True, st, ed))
            continue
        ch = v
        try:
            pi = pinyin(ch, style=Style.INITIALS, strict=False)  # type: ignore
            pf = pinyin(ch, style=Style.FINALS, strict=False)    # type: ignore
            pt = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)  # type: ignore
            init = pi[0][0] if pi and pi[0] else ""
            fin = pf[0][0] if pf and pf[0] else ""
            t0 = pt[0][0] if pt and pt[0] else ""
            tone = t0[-1] if t0 and t0[-1].isdigit() else "5"

            if init:
                seq.append(Phoneme(init, "zh", True, False, st, ed))
                if fin:
                    seq.append(Phoneme(fin, "zh", False, False, st, ed))
            else:
                if fin:
                    seq.append(Phoneme(fin, "zh", True, False, st, ed))
                else:
                    seq.append(Phoneme(ch, "zh", True, False, st, ed))
            seq.append(Phoneme(tone, "zh", False, True, st, ed))
        except Exception:
            seq.append(Phoneme(ch, "zh", True, True, st, ed))
    return seq


def _get_word_boundaries(text: str) -> List[Tuple[int, int, str]]:
    bounds: List[Tuple[int, int, str]] = []
    i, n = 0, len(text)
    while i < n:
        if not (text[i].isalnum() or _is_cjk(text[i])):
            i += 1
            continue
        s = i
        if _is_cjk(text[i]):
            i += 1
        else:
            low = text[i].islower()
            while i < n and text[i].isalnum():
                if text[i].isupper() and low and i > s:
                    break
                low = text[i].islower()
                i += 1
        bounds.append((s, i, text[s:i]))
    return bounds


def extract_diff_pairs(wrong: str, right: str) -> List[Tuple[str, str]]:
    wb, rb = _get_word_boundaries(wrong), _get_word_boundaries(right)
    wtok = [b[2] for b in wb]
    rtok = [b[2] for b in rb]
    m = SequenceMatcher(None, wtok, rtok)

    pairs: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == "replace":
            w_frag = wrong[wb[i1][0]:wb[i2 - 1][1]] if i2 > i1 else ""
            r_frag = right[rb[j1][0]:rb[j2 - 1][1]] if j2 > j1 else ""
            if w_frag and r_frag:
                pairs.append((w_frag, r_frag))

    if not pairs and wrong and right:
        pairs.append((wrong, right))

    out = []
    seen = set()
    for a, b in pairs:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


if HAS_NUMBA and HAS_NUMPY:
    @njit(cache=True)
    def _lev_min_dist(main_codes, sub_codes) -> float:
        n = len(sub_codes)
        m = len(main_codes)
        if n == 0:
            return 0.0
        if m == 0:
            return float(n)

        dp = np.zeros((n + 1, m + 1), dtype=np.float32)
        for i in range(1, n + 1):
            dp[i, 0] = float(i)
        for j in range(1, m + 1):
            dp[0, j] = 0.0

        for i in range(1, n + 1):
            si = sub_codes[i - 1]
            for j in range(1, m + 1):
                cost = 0.0 if si == main_codes[j - 1] else 1.0
                a = dp[i - 1, j] + 1.0
                b = dp[i, j - 1] + 1.0
                c = dp[i - 1, j - 1] + cost
                dp[i, j] = a if a <= b and a <= c else (b if b <= c else c)

        best = dp[n, 1]
        for j in range(2, m + 1):
            v = dp[n, j]
            if v < best:
                best = v
        return float(best)


class FastRAG:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.ph_to_code: Dict[str, int] = {}
        self.next_code = 1
        self.index = defaultdict(list)

    def _encode(self, p: str) -> int:
        v = self.ph_to_code.get(p)
        if v is None:
            v = self.next_code
            self.next_code += 1
            self.ph_to_code[p] = v
        return v

    def _encode_seq(self, seq: List[str]):
        if HAS_NUMPY:
            return np.array([self._encode(x) for x in seq], dtype=np.int32)
        return [self._encode(x) for x in seq]

    def add_hotwords(self, hotwords: Dict[str, List[Phoneme]]):
        for hw, phs in hotwords.items():
            if not phs:
                continue
            codes = self._encode_seq([p.value for p in phs])
            k = 2 if len(codes) >= 2 else 1
            for i in range(k):
                self.index[int(codes[i])].append((hw, codes))

    def _py_min_dist(self, main, sub) -> float:
        n, m = len(sub), len(main)
        if n == 0:
            return 0.0
        if m == 0:
            return float(n)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = float(i)
        for j in range(1, m + 1):
            dp[0][j] = 0.0
        for i in range(1, n + 1):
            si = sub[i - 1]
            for j in range(1, m + 1):
                cost = 0.0 if si == main[j - 1] else 1.0
                a = dp[i - 1][j] + 1.0
                b = dp[i][j - 1] + 1.0
                c = dp[i - 1][j - 1] + cost
                dp[i][j] = a if a <= b and a <= c else (b if b <= c else c)
        return min(dp[n][1:])

    def search(self, input_phs: List[Phoneme], top_k: int = 80) -> List[Tuple[str, float]]:
        if not input_phs:
            return []
        input_vals = [p.value for p in input_phs]
        main = self._encode_seq(input_vals)

        seen = set()
        cand: List[Tuple[str, object]] = []
        for p in input_phs:
            c = self.ph_to_code.get(p.value)
            if c is None:
                continue
            for hw, codes in self.index.get(c, []):
                if hw in seen:
                    continue
                seen.add(hw)
                cand.append((hw, codes))

        results: List[Tuple[str, float]] = []
        for hw, sub in cand:
            if len(sub) > len(input_phs) + 10:
                continue
            if HAS_NUMBA and HAS_NUMPY:
                dist = _lev_min_dist(main, sub)  # type: ignore
            else:
                dist = self._py_min_dist(main, sub)  # type: ignore
            score = 1.0 - (dist / max(1, len(sub)))
            if score >= self.threshold:
                results.append((hw, float(round(score, 3))))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class PhonemeCorrector:
    def __init__(
        self,
        threshold: float = 0.70,
        similar_threshold: Optional[float] = None,
        span_text_sim_min: float = 0.35,
        span_text_sim_min_short: float = 0.25,
        score_margin: float = 0.08,
        en_avg_sim_min: float = 0.68,
        en_single_sim_min: float = 0.86,
        en_need_one_strong: float = 0.80,
    ):
        self.threshold = threshold
        self.similar_threshold = (threshold - 0.15) if similar_threshold is None else similar_threshold
        self.span_text_sim_min = span_text_sim_min
        self.span_text_sim_min_short = span_text_sim_min_short
        self.score_margin = score_margin

        self.en_avg_sim_min = en_avg_sim_min
        self.en_single_sim_min = en_single_sim_min
        self.en_need_one_strong = en_need_one_strong

        self._lock = threading.Lock()
        self._entries: Dict[str, HotwordEntry] = {}
        self._zh_hotwords: Dict[str, List[Phoneme]] = {}
        self._rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.10)

        self._en_patterns_by_len: Dict[int, List[Tuple[str, Tuple[str, ...]]]] = {}
        self._en_max_len = 1

    @property
    def hotwords(self) -> Tuple[str, ...]:
        # Expose loaded hotword canonical forms for callers
        with self._lock:
            return tuple(self._entries.keys())

    def update_hotwords(self, text: str) -> int:
        lines = []
        for raw in text.splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

        entries: Dict[str, HotwordEntry] = {}
        zh_hotwords: Dict[str, List[Phoneme]] = {}
        en_patterns_by_len: Dict[int, List[Tuple[str, Tuple[str, ...]]]] = defaultdict(list)
        en_max_len = 1

        for ln in lines:
            e = _parse_hotword_line(ln)
            if e is None:
                continue
            entries[e.word] = e

            if _is_ascii_word(e.word):
                toks = [t for (t, lang, _, _) in _tokenize_with_offsets(e.word) if lang in ("en", "num")]
                if not toks:
                    continue
                base = tuple(toks)
                en_patterns_by_len[len(base)].append((e.word, base))
                en_max_len = max(en_max_len, len(base))

                more: List[Tuple[str, ...]] = []
                for i, tk in enumerate(base):
                    if tk == "overflow":
                        more.append(base[:i] + ("over", "flow") + base[i + 1:])
                for v in more:
                    en_patterns_by_len[len(v)].append((e.word, v))
                    en_max_len = max(en_max_len, len(v))
                continue

            phs = get_phoneme_info(e.word)
            if e.py_hint:
                try:
                    phs = _phonemes_from_py_hint(e.py_hint)
                except Exception:
                    pass
            if phs:
                zh_hotwords[e.word] = phs

        rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.10)
        rag.add_hotwords(zh_hotwords)

        with self._lock:
            self._entries = entries
            self._zh_hotwords = zh_hotwords
            self._rag = rag
            self._en_patterns_by_len = dict(en_patterns_by_len)
            self._en_max_len = en_max_len

        return len(entries)

    def load_hotwords_file(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return self.update_hotwords(f.read())

    def _span_text_gate_zh(self, e: HotwordEntry, span: str) -> bool:
        sim = _text_sim(span, e.word)
        th = self.span_text_sim_min_short if len(e.word) <= 3 else self.span_text_sim_min
        if e.min_text_sim is not None:
            th = e.min_text_sim
        return sim >= th

    def _passes_context_zh(self, e: HotwordEntry, text: str, st: int, ed: int) -> bool:
        w = e.word
        if st + len(w) <= len(text) and text[st:st + len(w)] == w:
            return False

        if e.min_span_len > 1 and (ed - st) < e.min_span_len:
            return False

        if e.prefix:
            if st == 0:
                return False
            prev = text[st - 1]
            if not _is_punct_or_space(prev):
                ok = False
                for p in e.prefix:
                    if text[max(0, st - len(p)):st] == p:
                        ok = True
                        break
                if not ok:
                    return False

        if e.suffix:
            if ed >= len(text):
                return e.allow_eos
            nxt = text[ed]
            if _is_punct_or_space(nxt):
                return True
            for s in e.suffix:
                if text.startswith(s, ed):
                    return True
            return False

        return True

    def _first_token_min(self, pat0: str) -> float:
        L = len(pat0)
        if L <= 2:
            return 0.45
        if L == 3:
            return 0.60
        return 0.70

    def _find_en_matches(self, text: str) -> List[MatchResult]:
        if not self._en_patterns_by_len:
            return []
        toks = _tokenize_with_offsets(text)
        if not toks:
            return []

        matches: List[MatchResult] = []
        i = 0
        n = len(toks)
        while i < n:
            if toks[i][1] not in ("en", "num"):
                i += 1
                continue
            j = i
            while j < n and toks[j][1] in ("en", "num"):
                j += 1
            seg = toks[i:j]
            vals = [x[0] for x in seg]

            p = 0
            while p < len(vals):
                best: Optional[Tuple[float, int, str, int, int]] = None
                for L in range(min(self._en_max_len, len(vals) - p), 0, -1):
                    plist = self._en_patterns_by_len.get(L)
                    if not plist:
                        continue

                    st = seg[p][2]
                    ed = seg[p + L - 1][3]
                    span_raw = text[st:ed]

                    if _span_has_cjk(span_raw):
                        continue
                    if len(_en_tokens(span_raw)) != L:
                        continue
                    if len(span_raw) > 48:
                        continue

                    span_vals = vals[p:p + L]
                    for canon, pat in plist:
                        sims = [_text_sim(span_vals[k], pat[k]) for k in range(L)]
                        avg = sum(sims) / L

                        if L == 1:
                            if avg < self.en_single_sim_min:
                                continue
                        else:
                            if sims[0] < self._first_token_min(pat[0]):
                                continue
                            if avg < self.en_avg_sim_min:
                                continue
                            if max(sims) < self.en_need_one_strong:
                                continue

                        if st + len(canon) <= len(text) and text[st:st + len(canon)] == canon:
                            continue

                        sc = 1.2 + avg
                        if best is None or sc > best[0]:
                            best = (float(sc), L, canon, st, ed)

                if best is not None:
                    sc, L, canon, st, ed = best
                    matches.append(MatchResult(st, ed, sc, canon))
                    p += L
                else:
                    p += 1
            i = j

        return matches

    def _find_zh_matches(self, text: str, in_phs: List[Phoneme]) -> Tuple[List[MatchResult], List[Tuple[str, str, float]]]:
        if not self._zh_hotwords:
            return [], []
        processed = [p.info for p in in_phs]
        processed5 = [x[:5] for x in processed]

        fast_res = self._rag.search(in_phs, top_k=120)

        matches: List[MatchResult] = []
        similars: List[Tuple[str, str, float]] = []
        search_thresh = min(self.threshold, self.similar_threshold) - 0.10

        for hw, _ in fast_res:
            phs = self._zh_hotwords.get(hw)
            if not phs:
                continue
            e = self._entries.get(hw)
            if e is None:
                continue

            hw_cmp = [p.info[:5] for p in phs]
            found = fuzzy_substring_search_constrained(hw_cmp, processed5, threshold=search_thresh)

            for score, s_idx, e_idx in found:
                st = processed[s_idx][5]
                ed = processed[e_idx - 1][6]
                if st >= ed:
                    continue
                span = text[st:ed]

                if not self._passes_context_zh(e, text, st, ed):
                    continue

                if not self._span_text_gate_zh(e, span):
                    if score >= self.similar_threshold:
                        similars.append((span, hw, float(score)))
                    continue

                if score + self.score_margin >= self.threshold:
                    matches.append(MatchResult(st, ed, float(score), hw))
                elif score >= self.similar_threshold:
                    similars.append((span, hw, float(score)))

        similars.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
        return matches, similars[:10]

    def _resolve_and_replace(self, text: str, matches: List[MatchResult]):
        matches.sort(key=lambda x: (x.score, x.end - x.start), reverse=True)
        keep: List[MatchResult] = []
        occupied: List[Tuple[int, int]] = []

        for m in matches:
            if any(not (m.end <= a or m.start >= b) for a, b in occupied):
                continue
            if text[m.start:m.end] == m.hotword:
                continue
            keep.append(m)
            occupied.append((m.start, m.end))

        keep.sort(key=lambda x: x.start, reverse=True)
        out = list(text)
        applied: List[Tuple[str, str, float]] = []
        for m in keep:
            src = text[m.start:m.end]
            dst = m.hotword
            out[m.start:m.end] = list(dst)
            applied.append((src, dst, m.score))

        return "".join(out), applied

    def correct(self, text: str) -> CorrectionResult:
        in_phs = get_phoneme_info(text)
        if not in_phs:
            return CorrectionResult(text, [], [])
        with self._lock:
            if not self._entries:
                return CorrectionResult(text, [], [])
            en_matches = self._find_en_matches(text)
            zh_matches, sims = self._find_zh_matches(text, in_phs)
            all_matches = en_matches + zh_matches
        new_text, applied = self._resolve_and_replace(text, all_matches)
        return CorrectionResult(new_text, applied, sims)


class RectificationRAG:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self._lock = threading.Lock()
        self._frags: List[Tuple[str, str, List[Tuple]]] = []

    @property
    def records(self) -> Tuple[Tuple[str, str], ...]:
        # Return stored (wrong,right) fragments for quick presence checks
        with self._lock:
            return tuple((w, r) for w, r, _ in self._frags)

    def load_rectify_text(self, text: str):
        frags: List[Tuple[str, str, List[Tuple]]] = []
        for block in text.split("---"):
            lines = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith("#")]
            if len(lines) < 2:
                continue
            wrong, right = lines[0], lines[1]
            for w_frag, r_frag in extract_diff_pairs(wrong, right):
                w_phs = [p.info[:5] for p in get_phoneme_info(w_frag)]
                if w_phs:
                    frags.append((w_frag, r_frag, w_phs))
        with self._lock:
            self._frags = frags

    def load_rectify_file(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            self.load_rectify_text(f.read())

    def search(self, text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        in_phs = [p.info[:5] for p in get_phoneme_info(text)]
        if not in_phs:
            return []
        with self._lock:
            fr = list(self._frags)

        hits: List[Tuple[str, str, float]] = []
        for w, r, w_phs in fr:
            found = fuzzy_substring_search_constrained(w_phs, in_phs, threshold=0.0)
            if not found:
                continue
            sc = max(x[0] for x in found)
            if sc >= self.threshold:
                hits.append((w, r, float(round(sc, 3))))

        hits.sort(key=lambda x: x[2], reverse=True)
        return hits[:top_k]

    def apply_corrections(self, text: str, top_k: int = 5) -> Tuple[str, List[Tuple[str, str, float]]]:
        matches = self.search(text, top_k=top_k)
        if not matches:
            return text, []
        out = text
        applied: List[Tuple[str, str, float]] = []
        for w, r, s in matches:
            if w and w in out and w != r:
                out = out.replace(w, r, 1)
                applied.append((w, r, s))
        return out, applied
