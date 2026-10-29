#!/usr/bin/env python3
"""Compares ZXing (reference) and rxing results of the barcode probe.

Usage: compare.py <zxing.jsonl> <rxing.jsonl>
"""

import json
import sys
from collections import Counter


def upce_to_upca(upce):
    """Expands an 8-digit UPC-E number to its 12-digit UPC-A form."""
    system, digits, check = upce[0], upce[1:7], upce[7]
    last = digits[5]
    if last in "012":
        middle = digits[0:2] + last + "0000" + digits[2:5]
    elif last == "3":
        middle = digits[0:3] + "00000" + digits[3:5]
    elif last == "4":
        middle = digits[0:4] + "00000" + digits[4]
    else:
        middle = digits[0:5] + "0000" + last
    return system + middle + check


def normalize(result):
    fmt = result["format"].lower().replace(" ", "").replace("-", "")
    text = result["text"]
    # ZXing-cpp reports UPC-E in its expanded EAN-13 form
    if fmt == "upce" and len(text) == 8:
        text = "0" + upce_to_upca(text)
    # ZXing-cpp keeps Codabar start and stop characters, rxing drops them (and its option to keep
    # them returns character indices instead of characters, an rxing bug)
    if fmt == "codabar" and len(text) >= 2 and text[0] in "ABCD" and text[-1] in "ABCD":
        text = text[1:-1]
    # ZXing reports UPC-A as EAN-13 with a leading zero
    if fmt == "upca":
        fmt, text = "ean13", "0" + text
    return fmt, text


def load(path):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        rows[row["file"]] = row
    return rows


def main():
    zxing, rxing = load(sys.argv[1]), load(sys.argv[2])
    outcome = Counter()
    by_variant = {}
    details = []
    for file in sorted(zxing):
        z = Counter(normalize(r) for r in zxing[file]["results"])
        x = Counter(normalize(r) for r in rxing[file]["results"])
        variant = "photo" if file.startswith("photo_") else file.split("_")[1]
        if z == x:
            kind = "same" if z else "both none"
        elif not x:
            kind = "rxing missed"
        elif not z:
            kind = "rxing extra"
        elif z - x:
            kind = "rxing missed some"
        else:
            kind = "rxing extra some"
        outcome[kind] += 1
        by_variant.setdefault(variant, Counter())[kind] += 1
        if kind not in ("same", "both none"):
            details.append(f"{file}: zxing={sorted(z)} rxing={sorted(x)}")

    ms_z = sum(r["ms"] for r in zxing.values())
    ms_x = sum(r["ms"] for r in rxing.values())
    print(f"images: {len(zxing)}")
    for kind, count in outcome.most_common():
        print(f"  {kind}: {count}")
    print("\nby variant:")
    for variant, counts in sorted(by_variant.items()):
        print(f"  {variant}: {dict(counts)}")
    print(f"\ndecoding time: zxing {ms_z:.0f} ms, rxing {ms_x:.0f} ms ({ms_x / ms_z:.1f}x)")
    print("\ndifferences:")
    for line in details:
        print("  " + line)


if __name__ == "__main__":
    main()
