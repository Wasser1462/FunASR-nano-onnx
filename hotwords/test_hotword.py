from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from hotword_corrector import PhonemeCorrector, RectificationRAG


@dataclass(frozen=True)
class TestCase:
    name: str
    text: str
    expect: Optional[str] = None
    must_contain: Tuple[str, ...] = ()
    must_not_contain: Tuple[str, ...] = ()
    expect_hotword_change: Optional[bool] = None
    expect_history_change: Optional[bool] = None
    note: str = ""


def _run_one(corrector: PhonemeCorrector, rectifier: RectificationRAG, tc: TestCase, history_top_k: int = 3):
    r1 = corrector.correct(tc.text)
    hotword_changed = (r1.text != tc.text)
    r2_text, r2_matches = rectifier.apply_corrections(r1.text, top_k=history_top_k)
    history_changed = (r2_text != r1.text)
    return {
        "original": tc.text,
        "after_hotword": r1.text,
        "after_history": r2_text,
        "hotword_changed": hotword_changed,
        "history_changed": history_changed,
        "hotword_matches": r1.matches,
        "hotword_similars": r1.similars,
        "history_matches": r2_matches,
    }


def _print_result(tc: TestCase, res: dict):
    print("\n" + "=" * 88)
    print(f"[{tc.name}]")
    if tc.note:
        print(f"NOTE: {tc.note}")
    print("-" * 88)
    print(f"Original:     {res['original']}")
    if res["hotword_changed"]:
        print(f"Hotword out:  {res['after_hotword']}")
        if res["hotword_matches"]:
            print(f"  Hotword matches ({len(res['hotword_matches'])}):")
            for w, r, s in res["hotword_matches"]:
                print(f"    - '{w}' => '{r}' (score={s:.3f})")
    else:
        print("Hotword out:  (no changes)")
        if res["hotword_similars"]:
            sims = res["hotword_similars"][:3]
            print(f"  Similar (below threshold, top {len(sims)}):")
            for w, r, s in sims:
                print(f"    - '{w}' => '{r}' (score={s:.3f})")

    if res["history_changed"]:
        print(f"History out:  {res['after_history']}")
        if res["history_matches"]:
            print(f"  History matches ({len(res['history_matches'])}):")
            for w, r, s in res["history_matches"]:
                print(f"    - '{w}' => '{r}' (score={s:.3f})")
    else:
        print("History out:  (no changes)")

    print(f"Final:        {res['after_history']}")
    if tc.expect is not None:
        print(f"Expect:       {tc.expect}")


def _check(tc: TestCase, res: dict) -> List[str]:
    errors: List[str] = []
    final_text = res["after_history"]

    if tc.expect is not None and final_text != tc.expect:
        errors.append("final != expect")

    for s in tc.must_contain:
        if s not in final_text:
            errors.append(f"missing required substring: {s!r}")

    for s in tc.must_not_contain:
        if s in final_text:
            errors.append(f"contains forbidden substring: {s!r}")

    if tc.expect_hotword_change is not None and res["hotword_changed"] != tc.expect_hotword_change:
        errors.append(f"hotword_changed != {tc.expect_hotword_change}")

    if tc.expect_history_change is not None and res["history_changed"] != tc.expect_history_change:
        errors.append(f"history_changed != {tc.expect_history_change}")

    return errors


def main():
    corrector = PhonemeCorrector(
        threshold=0.70,
        similar_threshold=0.55,
        span_text_sim_min=0.35,
        span_text_sim_min_short=0.25,
        score_margin=0.08,
    )
    rectifier = RectificationRAG(threshold=0.70)

    print("Loading hotwords and correction history...")
    n_hw = corrector.load_hotwords_file("hotwords.txt")
    rectifier.load_rectify_file("hot-rectify.txt")
    print(f"Loaded hotwords: {n_hw}")
    print("Loaded history:  (from hot-rectify.txt)")
    print()

    tests: List[TestCase] = []

    tests += [
        TestCase(
            name="EN-1 OpenAI/ChatGPT/TensorFlow/PyTorch/NumPy (spaced)",
            text="I tried open ai and chat gpt today; also tested tensor flow, pie torch, and num pie.",
            must_contain=("OpenAI", "ChatGPT", "TensorFlow", "PyTorch", "NumPy"),
            must_not_contain=("open ai", "chat gpt", "tensor flow", "pie torch", "num pie"),
        ),
        TestCase(
            name="EN-2 GitHub + StackOverflow (split tokens)",
            text="I found the answer on git hub and stack over flow.",
            must_contain=("GitHub", "StackOverflow"),
            must_not_contain=("git hub", "stack over flow", "TensorFlow"),
            note="Should not convert 'over flow' into TensorFlow (partial match protection).",
        ),
        TestCase(
            name="EN-3 Avoid false match: over flow != TensorFlow",
            text="The buffer may over flow if you ignore bounds.",
            must_not_contain=("TensorFlow",),
            expect_hotword_change=False,
            note="Negative sample: do NOT replace 'over flow' with TensorFlow.",
        ),
        TestCase(
            name="EN-4 Casing/punctuation noise",
            text="open ai, chat-gpt? tensor.flow! pie-torch & num.pie",
            must_contain=("OpenAI", "ChatGPT", "TensorFlow", "PyTorch", "NumPy"),
        ),
    ]

    tests += [
        TestCase(
            name="CN-1 麦当劳/肯德基",
            text="我想去吃买当劳和肯得鸡",
            expect="我想去吃麦当劳和肯德基",
        ),
        TestCase(
            name="CN-2 北京大学/清华大学",
            text="我考不上北京大血和清华大血",
            expect="我考不上北京大学和清华大学",
        ),
        TestCase(
            name="CN-3 机器学习/人工智能",
            text="人工智囊和机器学系是热门领域",
            expect="人工智能和机器学习是热门领域",
            note="This also checks 'no swallowing char' gate for zh hotwords.",
        ),
        TestCase(
            name="CN-4 Negative: prevent swallowing char in 公司",
            text="open ai和open ai都是人工智能公司",
            must_contain=("OpenAI", "人工智能公司"),
            must_not_contain=("人工智能司",),
            note="Negative: should not turn '人工智能公司' into '人工智能司'.",
        ),
    ]

    tests += [
        TestCase(
            name="CN-5 多音字提示：常安/长按 -> 长安 (brand)",
            text="常安汽车和重庆长按汽车有限公司是两家不同的公司",
            must_contain=("长安汽车", "重庆长安汽车有限公司"),
            must_not_contain=("常安汽车", "重庆长按汽车有限公司"),
            note="Requires hotwords.txt containing 长安|chang2 an and 重庆|chong qing (or similar).",
        ),
        TestCase(
            name="CN-6 Negative: '长按' as verb should not be changed",
            text="请长按电源键三秒钟重启设备",
            must_contain=("长按",),
            must_not_contain=("长安",),
            expect_hotword_change=False,
            note="Negative sample: do not rewrite UI instruction '长按'.",
        ),
    ]

    tests += [
        TestCase(
            name="MIX-1 Mixed EN/CN tech",
            text="我今天尝试了open AI和Chat gpt,还测试了tensor flow, pie torch, 和 num pie",
            must_contain=("OpenAI", "ChatGPT", "TensorFlow", "PyTorch", "NumPy"),
        ),
        TestCase(
            name="MIX-2 Mixed with GitHub/StackOverflow",
            text="我使用git hub和stack over flow来学习java script",
            must_contain=("GitHub", "StackOverflow", "JavaScript"),
            must_not_contain=("TensorFlow",),
        ),
    ]

    tests += [
        TestCase(
            name="HIST-1 History fragment: 实践->时间, 污点->五点",
            text="开饭实践早上九点到晚上污点",
            must_contain=("开饭时间", "晚上五点"),
            must_not_contain=("开饭实践", "晚上污点"),
            expect_hotword_change=False,
            expect_history_change=True,
            note="This should be corrected by history fragments, not hotwords.",
        ),
        TestCase(
            name="HIST-2 History + hotword",
            text="开饭实践早上九点至下午污点，我想去吃买当劳",
            must_contain=("开饭时间", "下午五点", "麦当劳"),
            must_not_contain=("开饭实践", "下午污点", "买当劳"),
            note="History fixes time; hotword fixes brand.",
        ),
    ]

    total = len(tests)
    ok = 0

    for tc in tests:
        res = _run_one(corrector, rectifier, tc, history_top_k=3)
        _print_result(tc, res)
        errs = _check(tc, res)

        if errs:
            print("\n[FAIL]")
            for e in errs:
                print("  -", e)
            cand = rectifier.search(res["after_hotword"], top_k=3)
            if cand:
                print("\nHistory candidates (debug):")
                for w, r, s in cand:
                    print(f"  - '{w}' => '{r}' (score={s:.3f})")
        else:
            print("\n[OK]")
            ok += 1

    print("\n" + "=" * 88)
    print("Summary")
    print("-" * 88)
    print(f"Total: {total}")
    print(f"OK:    {ok}")
    print(f"FAIL:  {total - ok}")
    print("=" * 88)

    if ok != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
