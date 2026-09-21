"""LoCoMo 官方评分口径：normalize → Porter stemming → token-level（多重集）F1。

与官方实现（snap-research/locomo task_eval/evaluation.py）逐项对齐：
- normalize_answer：去逗号 → 小写 → 去标点 → 去冠词/and（a|an|the|and，词边界）→ 空白归一
- f1_token_level：normalize 后按空白分词，Porter stem，Counter 多重集 F1
- 按类别差异（score_sample）：
    single-hop(4)/temporal(2)/open-domain(3) 直接 token-level F1
    multi-hop(1) 预测与答案按逗号拆成子答案，先 max 后 mean（部分给分）
    open-domain(3) 答案先取 ';' 首段
    adversarial(5) 输出含 "no information available"/"not mentioned" 记 1，否则 0

Porter stemmer 为纯 Python 紧凑实现，行为逐条对齐 nltk.stem.PorterStemmer
默认模式（NLTK_EXTENSIONS，即官方评测所用）。本模块零第三方依赖。
"""
import re
import string
from collections import Counter

CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain",
              4: "single_hop", 5: "adversarial"}
CATEGORY_NUM = {v: k for k, v in CATEGORIES.items()}


class PorterStemmer:
    """Porter (1980) suffix-stripping stemmer，对齐 nltk 默认 NLTK_EXTENSIONS 模式。"""

    _vowels = frozenset("aeiou")
    _irregular = {v: k for k, vs in {
        "sky": ["sky", "skies"], "die": ["dying"], "lie": ["lying"],
        "tie": ["tying"], "news": ["news"], "inning": ["innings", "inning"],
        "outing": ["outings", "outing"], "canning": ["cannings", "canning"],
        "howe": ["howe"], "proceed": ["proceed"], "exceed": ["exceed"],
        "succeed": ["succeed"]}.items() for v in vs}

    def _cv(self, word: str) -> str:
        """每个字符分类为 c/v（y 视前一字符而定，与 nltk _is_consonant 等价）。"""
        flags = []
        for i, ch in enumerate(word):
            if ch in self._vowels:
                flags.append("v")
            elif ch == "y":
                flags.append("c" if i == 0 or flags[i - 1] == "v" else "v")
            else:
                flags.append("c")
        return "".join(flags)

    def _m(self, stem: str) -> int:
        """measure m：cv 序列中 "vc" 出现次数。"""
        return self._cv(stem).count("vc")

    def _doublec(self, word: str) -> bool:
        return (len(word) >= 2 and word[-1] == word[-2]
                and self._cv(word)[-1] == "c")

    def _cvc(self, word: str) -> bool:
        if len(word) >= 3:
            return (self._cv(word)[-3:] == "cvc" and word[-1] not in "wxy")
        if len(word) == 2:  # NLTK_EXTENSIONS 的 2 字母规则
            return self._cv(word) == "vc"
        return False

    def stem(self, word: str) -> str:
        w = word.lower()
        if w in self._irregular:
            return self._irregular[w]
        if len(word) <= 2:
            return w
        w = self._step1a(w)
        w = self._step1b(w)
        w = self._step1c(w)
        w = self._step2(w)
        w = self._step3(w)
        w = self._step4(w)
        w = self._step5a(w)
        w = self._step5b(w)
        return w

    def _step1a(self, w: str) -> str:
        if w.endswith("ies") and len(w) == 4:
            return w[:-3] + "ie"
        if w.endswith("sses"):
            return w[:-4] + "ss"
        if w.endswith("ies"):
            return w[:-3] + "i"
        if w.endswith("ss"):
            return w
        if w.endswith("s"):
            return w[:-1]
        return w

    def _step1b(self, w: str) -> str:
        if w.endswith("ied"):
            return w[:-3] + ("ie" if len(w) == 4 else "i")
        if w.endswith("eed"):
            stem = w[:-3]
            return stem + "ee" if self._m(stem) > 0 else w
        inter = None
        for suf in ("ed", "ing"):
            if w.endswith(suf):
                t = w[: -len(suf)]
                if "v" in self._cv(t):
                    inter = t
                    break
        if inter is None:
            return w
        if inter.endswith("at"):
            return inter[:-2] + "ate"
        if inter.endswith("bl"):
            return inter[:-2] + "ble"
        if inter.endswith("iz"):
            return inter[:-2] + "ize"
        if self._doublec(inter):  # *d and not (*L|*S|*Z) → 单字母；命中即终止
            return inter[:-1] if inter[-1] not in "lsz" else inter
        if self._m(inter) == 1 and self._cvc(inter):
            return inter + "e"
        return inter

    def _step1c(self, w: str) -> str:
        if w.endswith("y"):
            stem = w[:-1]
            if len(stem) > 1 and self._cv(stem)[-1] == "c":
                return stem + "i"
        return w

    def _step2(self, w: str) -> str:
        if w.endswith("alli") and self._m(w[:-4]) > 0:  # NLTK_EXTENSIONS 前置规则
            return self._step2(w[:-4] + "al")
        rules = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
                 ("anci", "ance"), ("izer", "ize"), ("bli", "ble"),
                 ("alli", "al"), ("entli", "ent"), ("eli", "e"),
                 ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
                 ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
                 ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
                 ("iviti", "ive"), ("biliti", "ble"), ("fulli", "ful"),
                 ("logi", "log")]
        for suf, rep in rules:
            if w.endswith(suf):
                if suf == "logi":  # 条件用 word[:-3]（= stem+"l"）
                    return w[:-4] + rep if self._m(w[:-3]) > 0 else w
                stem = w[: -len(suf)]
                return stem + rep if self._m(stem) > 0 else w
        return w

    def _step3(self, w: str) -> str:
        for suf, rep in [("icate", "ic"), ("ative", ""), ("alize", "al"),
                         ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
                         ("ness", "")]:
            if w.endswith(suf):
                stem = w[: -len(suf)]
                return stem + rep if self._m(stem) > 0 else w
        return w

    def _step4(self, w: str) -> str:
        for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant",
                    "ement", "ment", "ent"):
            if w.endswith(suf):
                stem = w[: -len(suf)]
                return stem if self._m(stem) > 1 else w
        if w.endswith("ion"):
            stem = w[:-3]
            return stem if (self._m(stem) > 1 and stem[-1] in "st") else w
        for suf in ("ou", "ism", "ate", "iti", "ous", "ive", "ize"):
            if w.endswith(suf):
                stem = w[: -len(suf)]
                return stem if self._m(stem) > 1 else w
        return w

    def _step5a(self, w: str) -> str:
        if w.endswith("e"):
            stem = w[:-1]
            if self._m(stem) > 1:
                return stem
            if self._m(stem) == 1 and not self._cvc(stem):
                return stem
        return w

    def _step5b(self, w: str) -> str:
        if w.endswith("ll") and self._m(w[:-1]) > 1:
            return w[:-1]
        return w


_STEMMER = PorterStemmer()

_ARTICLES_RE = re.compile(r"\b(a|an|the|and)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.replace(",", "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def f1_token_level(prediction: str, gold: str) -> float:
    """官方 f1_score：normalize → stem → 空白分词 → 多重集 F1。"""
    pt = [_STEMMER.stem(w) for w in normalize_answer(prediction).split()]
    gt = [_STEMMER.stem(w) for w in normalize_answer(gold).split()]
    num_same = sum((Counter(pt) & Counter(gt)).values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pt)
    rec = num_same / len(gt)
    return 2 * prec * rec / (prec + rec)


def f1_multi(prediction: str, gold: str) -> float:
    """官方 f1（multi-hop 用）：按逗号拆子答案，每个 gold 取最佳 pred，再平均。"""
    preds = [p.strip() for p in prediction.split(",")]
    golds = [g.strip() for g in gold.split(",")]
    return sum(max(f1_token_level(p, g) for p in preds)
               for g in golds) / len(golds)


def score_sample(category: int, prediction: str, answer) -> float:
    """按 LoCoMo 官方类别规则打一分。"""
    answer = str(answer)
    if category == 3:  # open-domain：答案取 ';' 首段
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_token_level(prediction, answer)
    if category == 1:
        return f1_multi(prediction, answer)
    if category == 5:
        low = prediction.lower()
        return 1.0 if ("no information available" in low
                       or "not mentioned" in low) else 0.0
    raise ValueError(f"未知类别: {category}")
