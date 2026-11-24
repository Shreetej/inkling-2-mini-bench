"""Task suite for the Inkling 2 Mini comparison.

Every task is graded by code, not by another model:
  code / debug  -> extract the Python block, run hidden asserts in a subprocess
  instr         -> programmatic constraint checks
  reason / fact -> normalised answer match
  abstain       -> questions about things that do not exist; pass = the model says it doesn't know
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

CODE_SUFFIX = ("\n\nReturn only the complete Python implementation in a single ```python code block. "
               "Do not include tests or example usage.")


@dataclass
class Task:
    id: str
    cat: str
    prompt: str
    check: Callable[[str], bool] = field(repr=False)


# ---------------------------------------------------------------- helpers
def code_block(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def run_tests(tests: str, prelude: str = "") -> Callable[[str], bool]:
    """`prelude` is the original snippet for debug tasks, so a fix may use its module-level constants."""
    def check(answer: str) -> bool:
        src = prelude + "\n\n" + code_block(answer) + "\n\n" + tests + "\nprint('ALL_OK')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0 and "ALL_OK" in r.stdout
    return check


def norm(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.strip().lower())


def final_line(s: str) -> str:
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def contains(*needles: str) -> Callable[[str], bool]:
    return lambda a: any(norm(n) in norm(a) for n in needles)


def number_is(n: int) -> Callable[[str], bool]:
    def check(a: str) -> bool:
        nums = re.findall(r"-?\d+", a.replace(",", ""))
        return bool(nums) and int(nums[-1]) == n and len(norm(a)) < 40
    return check


ABSTAIN = ["not aware", "no record", "don't know", "do not know", "couldn't find", "could not find",
           "not familiar", "unable to find", "no information", "does not appear", "doesn't appear",
           "may not exist", "might not exist", "does not exist", "doesn't exist", "fictional",
           "not a real", "cannot verify", "can't verify", "no reliable", "not able to find",
           "unable to verify", "no known", "i'm not sure", "i am not sure", "not find any",
           "no widely", "not widely known", "unaware", "i don't have", "i do not have",
           "can't find", "cannot find", "not something i", "no such", "not recognize",
           "don't recognize", "do not recognize", "unable to confirm", "can't confirm", "cannot confirm",
           "not able to verify", "not able to confirm", "isn't a", "is not a known", "no evidence",
           "not a widely known", "never been awarded", "no entry for", "not able to locate", "unable to locate",
           "don't appear", "do not appear"]


def abstains(a: str) -> bool:
    t = norm(a).replace("’", "'")
    return any(m in t for m in ABSTAIN)


# ---------------------------------------------------------------- coding (16)
CODE: list[tuple[str, str, str]] = [
    ("merge_intervals",
     "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals. "
     "Intervals that touch (e.g. [1,2] and [2,3]) count as overlapping. Return them sorted by start. "
     "The input may be unsorted or empty.",
     """assert merge_intervals([]) == []
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,2],[2,3]]) == [[1,3]]
assert merge_intervals([[5,7],[1,4],[0,0]]) == [[0,0],[1,4],[5,7]]
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]"""),
    ("lru_cache",
     "Write a class `LRU` with `__init__(self, capacity: int)`, `get(self, key) -> int` (returns -1 if missing) and "
     "`put(self, key, value) -> None`. When over capacity, evict the least recently used key. Both get and put "
     "count as a use. Both operations must be O(1).",
     """c = LRU(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c.put(3, 30); c.put(5, 5)
assert c.get(4) == -1 and c.get(3) == 30
d = LRU(1); d.put('a', 1); d.put('a', 2); assert d.get('a') == 2"""),
    ("parse_duration",
     "Write `parse_duration(s: str) -> int` that converts strings like '1h30m', '2d', '45s', '1d2h3m4s' to seconds. "
     "Units are d, h, m, s, each at most once, in that order. Raise ValueError for an empty string, "
     "unknown units, repeated or out-of-order units, or any other malformed input.",
     """assert parse_duration('45s') == 45
assert parse_duration('1h30m') == 5400
assert parse_duration('2d') == 172800
assert parse_duration('1d2h3m4s') == 93784
for bad in ['', '5', '1x', '1m1h', '1h1h', 'h', '1.5h', ' 1h']:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)"""),
    ("topo_sort",
     "Write `topo_sort(deps: dict[str, list[str]]) -> list[str]`. `deps[x]` lists the nodes that must come before x. "
     "Nodes may appear only inside dependency lists. Return a topological order; when several nodes are available, "
     "pick the alphabetically smallest first. Raise ValueError if there is a cycle.",
     """assert topo_sort({}) == []
assert topo_sort({'b': ['a'], 'c': ['a'], 'd': ['b', 'c']}) == ['a', 'b', 'c', 'd']
assert topo_sort({'z': [], 'y': [], 'x': ['z']}) == ['y', 'z', 'x']
assert topo_sort({'app': ['lib', 'log'], 'lib': ['core']}) == ['core', 'lib', 'log', 'app']
try:
    topo_sort({'a': ['b'], 'b': ['c'], 'c': ['a']})
except ValueError:
    pass
else:
    raise AssertionError('cycle')"""),
    ("eval_rpn",
     "Write `eval_rpn(tokens: list[str]) -> int` evaluating Reverse Polish Notation with + - * /. "
     "Operands are integers (possibly negative, e.g. '-3'). Division truncates toward zero.",
     """assert eval_rpn(['2','1','+','3','*']) == 9
assert eval_rpn(['4','13','5','/','+']) == 6
assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+']) == 22
assert eval_rpn(['-7','2','/']) == -3
assert eval_rpn(['7','-2','/']) == -3
assert eval_rpn(['-3']) == -3"""),
    ("longest_palindrome",
     "Write `longest_palindrome(s: str) -> str` returning the longest palindromic substring. If several have the "
     "same maximum length, return the one that starts earliest. Return '' for ''.",
     """assert longest_palindrome('') == ''
assert longest_palindrome('babad') == 'bab'
assert longest_palindrome('cbbd') == 'bb'
assert longest_palindrome('abc') == 'a'
assert longest_palindrome('forgeeksskeegfor') == 'geeksskeeg'
assert longest_palindrome('aaaa') == 'aaaa'"""),
    ("wildcard_match",
     "Write `wildcard_match(s: str, p: str) -> bool` where '?' matches exactly one character and '*' matches any "
     "sequence including empty. The whole string must match. It must handle 2,000-character inputs quickly.",
     """assert wildcard_match('aa', 'a') is False
assert wildcard_match('aa', '*') is True
assert wildcard_match('cb', '?a') is False
assert wildcard_match('adceb', '*a*b') is True
assert wildcard_match('acdcb', 'a*c?b') is False
assert wildcard_match('', '***') is True
assert wildcard_match('', '?') is False
import time; t = time.time()
assert wildcard_match('a' * 2000, '*' * 50 + 'b') is False
assert time.time() - t < 5"""),
    ("next_permutation",
     "Write `next_permutation(nums: list[int]) -> list[int]` returning a NEW list with the next lexicographic "
     "permutation. If nums is the last permutation, return it sorted ascending. Do not modify the input. "
     "Values may repeat.",
     """a = [1, 2, 3]
assert next_permutation(a) == [1, 3, 2] and a == [1, 2, 3]
assert next_permutation([3, 2, 1]) == [1, 2, 3]
assert next_permutation([1, 1, 5]) == [1, 5, 1]
assert next_permutation([1, 5, 1]) == [5, 1, 1]
assert next_permutation([2, 3, 1, 3, 3]) == [2, 3, 3, 1, 3]
assert next_permutation([]) == []"""),
    ("calc",
     "Write `calc(expr: str) -> float` evaluating arithmetic with + - * /, parentheses, unary minus, decimals and "
     "arbitrary spaces, using normal precedence (* and / before + and -, left-associative). "
     "Do not use eval, exec or compile.",
     """import math
assert math.isclose(calc('1 + 2 * 3'), 7)
assert math.isclose(calc('(1 + 2) * 3'), 9)
assert math.isclose(calc('-(2 + 3) * -2'), 10)
assert math.isclose(calc('8 / 4 / 2'), 1)
assert math.isclose(calc('10 - 4 - 3'), 3)
assert math.isclose(calc('2 * -3.5'), -7)
assert math.isclose(calc(' 3.25+ (4 -1)*2 '), 9.25)
import inspect
assert 'eval(' not in inspect.getsource(calc)"""),
    ("spiral",
     "Write `spiral(matrix: list[list[int]]) -> list[int]` returning elements in clockwise spiral order starting "
     "at the top-left. Handle empty and non-square matrices.",
     """assert spiral([]) == []
assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
assert spiral([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]
assert spiral([[1],[2],[3]]) == [1,2,3]
assert spiral([[1,2,3]]) == [1,2,3]"""),
    ("edit_distance",
     "Write `edit_distance(a: str, b: str) -> int` returning the Levenshtein distance (insert, delete, substitute "
     "each cost 1).",
     """assert edit_distance('', '') == 0
assert edit_distance('horse', 'ros') == 3
assert edit_distance('intention', 'execution') == 5
assert edit_distance('abc', '') == 3
assert edit_distance('kitten', 'sitting') == 3"""),
    ("dijkstra",
     "Write `shortest_paths(n: int, edges: list[tuple[int, int, int]], src: int) -> list[int]` for a DIRECTED graph "
     "with nodes 0..n-1 and non-negative weights (u, v, w). Return the shortest distance from src to every node, "
     "using -1 for unreachable nodes. There may be parallel edges and self-loops.",
     """assert shortest_paths(1, [], 0) == [0]
assert shortest_paths(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)], 0) == [0,3,1,4]
assert shortest_paths(3, [(1,0,1)], 0) == [0,-1,-1]
assert shortest_paths(3, [(0,1,5),(0,1,2),(1,1,0),(1,2,0)], 0) == [0,2,2]"""),
    ("sliding_median",
     "Write `sliding_median(nums: list[int], k: int) -> list[float]` returning the median of every window of size "
     "k, as floats. It must handle 100,000 numbers with k = 1,000 in under 5 seconds.",
     """assert sliding_median([1,3,-1,-3,5,3,6,7], 3) == [1.0,-1.0,-1.0,3.0,5.0,6.0]
assert sliding_median([1,2,3,4], 2) == [1.5,2.5,3.5]
assert sliding_median([5], 1) == [5.0]
import random, time
random.seed(7)
xs = [random.randint(-10**6, 10**6) for _ in range(100000)]
t = time.time(); out = sliding_median(xs, 1000)
assert time.time() - t < 5 and len(out) == 99001
assert out[0] == sorted(xs[:1000])[499] / 2 + sorted(xs[:1000])[500] / 2"""),
    ("decode_string",
     "Write `decode_string(s: str) -> str` that expands patterns like '3[a2[c]]' to 'accaccacc'. Counts can have "
     "several digits and patterns can nest. Plain letters may appear anywhere.",
     """assert decode_string('3[a]2[bc]') == 'aaabcbc'
assert decode_string('3[a2[c]]') == 'accaccacc'
assert decode_string('2[abc]3[cd]ef') == 'abcabccdcdcdef'
assert decode_string('10[x]') == 'x' * 10
assert decode_string('ab') == 'ab'"""),
    ("group_anagrams",
     "Write `group_anagrams(words: list[str]) -> list[list[str]]`. Sort each group alphabetically, and sort the "
     "groups by their first element. Duplicate words stay in the group.",
     """assert group_anagrams([]) == []
assert group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]
assert group_anagrams(['ab','ba','ab']) == [['ab','ab','ba']]"""),
    ("rate_limiter",
     "Write a class `SlidingWindowLimiter` with `__init__(self, limit: int, window: float)` and "
     "`allow(self, key: str, now: float) -> bool`. A request is allowed if fewer than `limit` requests for that key "
     "were ALLOWED in the half-open interval (now - window, now]. Rejected requests do not count. "
     "Keys are independent. `now` never decreases.",
     """r = SlidingWindowLimiter(2, 10)
assert r.allow('a', 0) and r.allow('a', 1)
assert not r.allow('a', 5)
assert r.allow('b', 5)
assert not r.allow('a', 9.99)
assert r.allow('a', 10)
assert not r.allow('a', 10.5)
assert r.allow('a', 11)"""),
]

# ---------------------------------------------------------------- debugging (5)
DEBUG: list[tuple[str, str, str]] = [
    ("tiered_bill",
     """This function should compute an electricity bill with marginal tiers: the first 100 kWh cost 0.10 each, the next 200 kWh (101-300) cost 0.15 each, and every kWh above 300 costs 0.20. A customer using 350 kWh should pay 10 + 30 + 10 = 50.0, but the function returns 70.0. Fix it.

```python
TIERS = [(100, 0.10), (300, 0.15), (float('inf'), 0.20)]

def bill(kwh: float) -> float:
    for limit, rate in TIERS:
        if kwh <= limit:
            return round(kwh * rate, 2)
```""",
     """assert bill(0) == 0
assert bill(50) == 5.0
assert bill(100) == 10.0
assert bill(250) == 32.5
assert bill(300) == 40.0
assert bill(350) == 50.0
assert bill(1000) == 180.0"""),
    ("binary_search",
     """This should return the index of `target` in the sorted list `xs`, or -1. It hangs on some inputs and misses some elements. Fix it.

```python
def find(xs: list[int], target: int) -> int:
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid
        elif xs[mid] > target:
            hi = mid
        else:
            return mid
    return -1
```""",
     """xs = [1, 3, 5, 7, 9, 11]
for i, x in enumerate(xs):
    assert find(xs, x) == i
for x in [0, 2, 4, 12]:
    assert find(xs, x) == -1
assert find([], 1) == -1
assert find([4], 4) == 0"""),
    ("mutable_default",
     """Tags leak between unrelated calls: `tag('a')` then `tag('b')` returns ['x', 'x'] for the second item. Fix it without changing the signature's meaning (callers may still pass their own list, which should be appended to and returned).

```python
def tag(item: str, tags: list[str] = []) -> list[str]:
    tags.append('x')
    return tags
```""",
     """assert tag('a') == ['x']
assert tag('b') == ['x']
mine = ['y']
assert tag('c', mine) == ['y', 'x'] and mine == ['y', 'x']"""),
    ("chunker",
     """`chunk([1,2,3,4,5], 2)` should return [[1,2],[3,4],[5]] but the last element is lost. It must also raise ValueError when size < 1. Fix it.

```python
def chunk(xs: list, size: int) -> list[list]:
    return [xs[i:i + size] for i in range(0, len(xs) - size + 1, size)]
```""",
     """assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]
assert chunk([], 3) == []
assert chunk([1], 5) == [[1]]
for bad in (0, -1):
    try:
        chunk([1], bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)"""),
    ("moving_average",
     """`moving_average([1,2,3,4,5], 3)` should return [2.0, 3.0, 4.0]. It returns the wrong values and is O(n*k); it should be O(n). Fix both.

```python
def moving_average(xs: list[float], k: int) -> list[float]:
    out = []
    total = 0
    for i in range(len(xs)):
        total += xs[i]
        if i >= k:
            total -= xs[i - k + 1]
        if i >= k - 1:
            out.append(total / k)
    return out
```""",
     """assert moving_average([1,2,3,4,5], 3) == [2.0, 3.0, 4.0]
assert moving_average([5,5,5], 1) == [5.0, 5.0, 5.0]
assert moving_average([1,2], 3) == []
import time
t = time.time(); r = moving_average(list(range(200000)), 5000)
assert time.time() - t < 2 and len(r) == 195001 and r[0] == 2499.5"""),
]

# ---------------------------------------------------------------- instruction following (9)
def _three_bullets(a: str) -> bool:
    lines = [ln for ln in a.strip().splitlines() if ln.strip()]
    return len(lines) == 3 and all(ln.startswith("- ") and len(ln[2:].split()) < 12 for ln in lines)


def _json_obj(a: str) -> bool:
    s = a.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        o = json.loads(s)
    except json.JSONDecodeError:
        return False
    return (isinstance(o, dict) and set(o) == {"name", "year", "tags"} and isinstance(o["name"], str)
            and isinstance(o["year"], int) and isinstance(o["tags"], list) and len(o["tags"]) == 3
            and all(isinstance(t, str) and t == t.lower() for t in o["tags"]))


def _no_e(a: str) -> bool:
    s = a.strip()
    return len(s.split()) >= 8 and "e" not in s.lower() and "\n" not in s


def _haiku(a: str) -> bool:
    lines = [ln.strip() for ln in a.strip().splitlines() if ln.strip()]
    text = " ".join(lines)
    return len(lines) == 3 and text == text.lower() and not re.search(r"[^\w\s']", text)


def _fifty(a: str) -> bool:
    return len(a.split()) == 50


def _title(a: str) -> bool:
    s = a.strip().strip('"')
    return ("\n" not in s and len(s.split()) <= 8 and "insulated" in s.lower() and "," not in s
            and re.sub(r"[^\w]+$", "", s).endswith("Kova"))


def _table(a: str) -> bool:
    rows = [ln.strip() for ln in a.strip().splitlines() if ln.strip().startswith("|")]
    if len(rows) != 5 or len(rows) != len([ln for ln in a.strip().splitlines() if ln.strip()]):
        return False
    body = [norm(r) for r in rows[2:]]
    return ("french" in body[0] and "bonjour" in body[0] and "spanish" in body[1] and "buenos dias" in body[1]
            and "german" in body[2] and "guten morgen" in body[2])


INSTR: list[tuple[str, str, Callable[[str], bool]]] = [
    ("three_bullets", "Give exactly 3 bullet points on why code review matters. Each line must start with '- ' and "
     "contain fewer than 12 words. Output nothing else.", _three_bullets),
    ("json_only", "Return only a JSON object (no prose, no code fence) describing the Eiffel Tower with exactly these "
     "keys: name (string), year (integer, year completed), tags (array of exactly 3 lowercase strings).", _json_obj),
    ("no_letter_e", "Write one sentence of at least 8 words about the ocean without using the letter 'e' anywhere. "
     "Output only the sentence.", _no_e),
    ("haiku", "Write a haiku about the Rust programming language. Exactly 3 lines, all lowercase, no punctuation at "
     "all. Output only the haiku.", _haiku),
    ("primes_done", "List the first 10 prime numbers separated by ', ' on one line, then write DONE on the next line. "
     "Output nothing else.",
     lambda a: a.strip() == "2, 3, 5, 7, 11, 13, 17, 19, 23, 29\nDONE"),
    ("fifty_words", "In exactly 50 words, explain why automated tests matter. Output only those 50 words.", _fifty),
    ("blue", "Output the word blue in uppercase 4 times, separated by single hyphens, and nothing else.",
     lambda a: a.strip() == "BLUE-BLUE-BLUE-BLUE"),
    ("product_title", "Write a product title for a water bottle: at most 8 words, must include the word "
     "'insulated', no commas, and it must end with the brand name Kova. Output only the title.", _title),
    ("md_table", "Translate 'good morning' into French, Spanish and German. Output only a markdown table with the "
     "header row 'Language | Phrase', a separator row, and one row per language in that order.", _table),
]

# ---------------------------------------------------------------- reasoning (6)
REASON: list[tuple[str, str, Callable[[str], bool]]] = [
    ("count_r", "How many times does the letter r appear in 'strawberry raspberry'? Reply with only the number.",
     number_is(6)),
    ("marbles", "Box A has twice as many marbles as box B. Box C has 7 fewer marbles than box A. The three boxes hold "
     "58 marbles in total. How many marbles are in box C? Reply with only the number.", number_is(19)),
    ("weekday", "What day of the week was 1 January 2026? Reply with only the day name.",
     lambda a: norm(final_line(a)).strip(".") == "thursday"),
    ("div_3_5", "How many integers from 1 to 1000 inclusive are divisible by 3 or by 5, but not by 15? "
     "Reply with only the number.", number_is(401)),
    ("sort_words", "Sort these words by length, then alphabetically within the same length: pear, fig, banana, kiwi, "
     "apple, date. Reply with only the words, comma-separated.",
     lambda a: re.sub(r"\s", "", norm(final_line(a))).strip(".") == "fig,date,kiwi,pear,apple,banana"),
    ("train", "A train departs at 09:47 and arrives at 14:12 the same day. How many minutes is the journey? "
     "Reply with only the number.", number_is(265)),
]

# ---------------------------------------------------------------- factual recall (12) + abstention (4)
FACT_SUFFIX = " Answer with only the answer, no explanation."
FACT: list[tuple[str, str, Callable[[str], bool]]] = [
    ("burkina", "What is the capital of Burkina Faso?", contains("ouagadougou")),
    ("margarita", "Who wrote the novel 'The Master and Margarita'?", contains("bulgakov")),
    ("tungsten", "What is the chemical symbol for tungsten?", lambda a: norm(a).strip(".") == "w"),
    ("westphalia", "In what year was the Peace of Westphalia signed?", number_is(1648)),
    ("rhodium", "What is the atomic number of rhodium?", number_is(45)),
    ("rite", "Who composed 'The Rite of Spring'?", contains("stravinsky")),
    ("weber", "What is the SI unit of magnetic flux?", contains("weber")),
    ("chernobyl", "In what year did the Chernobyl nuclear disaster happen?", number_is(1986)),
    ("titan", "Which planet does the moon Titan orbit?", contains("saturn")),
    ("solitude", "Who wrote 'One Hundred Years of Solitude'?", contains("marquez")),
    ("pydict", "In Python, what does len({1: 'a', True: 'b', 1.0: 'c'}) return?", number_is(1)),
    ("nauru", "What is the smallest country by area in Oceania that is a UN member?", contains("nauru")),
]
ABSTAIN_QS: list[tuple[str, str]] = [
    ("fake_abel", "In which year did the Norwegian mathematician Halvard Brennsvik win the Abel Prize?"),
    ("fake_novel", "Summarise the plot of the 1987 novel 'The Glass Cartographer' by Imelda Ross-Fenwick in two "
     "sentences."),
    ("fake_port", "What is the default TCP port of the Zephyrine message broker?"),
    ("fake_prize", "Who won the 1994 Kestrel Prize for marine biology?"),
]


def all_tasks() -> list[Task]:
    out: list[Task] = []
    out += [Task(f"code/{i}", "code", p + CODE_SUFFIX, run_tests(t)) for i, p, t in CODE]
    out += [Task(f"debug/{i}", "debug", p + "\n\nReturn only the fixed function in a single ```python code block.",
                 run_tests(t, code_block(p))) for i, p, t in DEBUG]
    out += [Task(f"instr/{i}", "instr", p, c) for i, p, c in INSTR]
    out += [Task(f"reason/{i}", "reason", p, c) for i, p, c in REASON]
    out += [Task(f"fact/{i}", "fact", p + FACT_SUFFIX, c) for i, p, c in FACT]
    out += [Task(f"abstain/{i}", "abstain", p, abstains) for i, p in ABSTAIN_QS]
    return out
