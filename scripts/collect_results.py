#!/usr/bin/env python3
# collects the summary lines of every verify_experiments.sh run into tables
# usage: scripts/collect_results.py [results_dir], default results/verify
# writes <results_dir>/summary.csv (every field) and summary.md (report tables)
import csv
import re
import sys
from pathlib import Path

# file name written by verify_experiments.sh
RUN_NAME = re.compile(
    r"^(?P<set>validation|test)_(?P<method>[a-z]+)_D(?P<D>\d+)_C(?P<C>\d+)"
    r"_rho(?P<rho>[0-9.]+)_tau(?P<tau>[0-9.]+)_T(?P<T>\d+)_adaptive(?P<adaptive>[01])$")

# summary lines of the output file, the same names after each group prefix
SUMMARY_FIELDS = [
    "MeanBeforeRecall@5", "MeanBeforeRecall@10", "MeanBeforeMAP@10",
    "MeanAfterRecall@5", "MeanAfterRecall@10", "MeanAfterMAP@10",
    "MeanMatchCountRecall@5", "MeanMatchCountRecall@10", "MeanMatchCountMAP@10",
    "AverageCandidateTimeMs", "AverageVerificationTimeMs", "AverageTotalTimeMs",
    "ValidHomographyRate", "AccurateHomographyRate1px", "AccurateHomographyRate3px",
    "AccurateHomographyRate5px", "MeanCornerErrorPx", "QueriesBelowC",
]
# "" is the summary of all queries
GROUPS = [("all", ""), ("illumination", "Illumination-"), ("viewpoint", "Viewpoint-")]
PARAMETERS = ["set", "method", "D", "distractors", "C", "rho", "tau", "T", "adaptive"]
SET_ORDER = {"validation": 0, "test": 1}
METHOD_ORDER = {"exact": 0, "lsh": 1, "hypercube": 2, "ivfflat": 3, "ivfpq": 4}


def read_fields(path):
    # "Key: value" lines, summary keys appear once per file
    fields = {}
    query_count = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            key, separator, value = line.partition(": ")
            if not separator:
                continue
            if key == "Query":
                query_count += 1
            fields[key] = value.strip()
    return fields, query_count


def read_log(path):
    # distractor count and wall time, empty if the log is missing
    distractors = wall_time = ""
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        # last match: a cache hit prints the line too
        found = re.findall(r"^using (\d+) MIRFlickr distractors", text, re.MULTILINE)
        if found:
            distractors = found[-1]
        found = re.findall(r"^wall time: (\d+) s", text, re.MULTILINE)
        if found:
            wall_time = found[-1]
    return distractors, wall_time


def collect(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*.txt")):
        name = RUN_NAME.match(path.stem)
        if not name:
            continue
        fields, query_count = read_fields(path)
        # verify writes the file at the end, the summary marks a finished run
        if "MeanAfterMAP@10" not in fields:
            print(f"warning: skipped incomplete {path.name}", file=sys.stderr)
            continue
        distractors, wall_time = read_log(path.with_suffix(".log"))
        parameters = dict(name.groupdict(), distractors=distractors)
        for group, prefix in GROUPS:
            # an empty group has no lines in the output
            queries = query_count if not prefix else fields.get(prefix + "Queries")
            if queries is None:
                continue
            row = dict(parameters, group=group, queries=str(queries), wall_time_s=wall_time)
            for field in SUMMARY_FIELDS:
                row[field] = fields.get(prefix + field, "")
            rows.append(row)

    # validation first, then method, D, C, rho, tau, T, adaptive, group
    group_order = {group: i for i, (group, _) in enumerate(GROUPS)}
    rows.sort(key=lambda row: (
        SET_ORDER[row["set"]], METHOD_ORDER.get(row["method"], 99), int(row["D"]),
        int(row["C"]), float(row["rho"]), float(row["tau"]), int(row["T"]),
        -int(row["adaptive"]), group_order[row["group"]]))
    return rows


def write_csv(rows, path):
    columns = PARAMETERS + ["group", "queries"] + SUMMARY_FIELDS + ["wall_time_s"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def milliseconds(value):
    # times to one decimal, the rest as printed by verify
    return f"{float(value):.1f}" if value else ""


def markdown_table(header, lines):
    text = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    text += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(text) + "\n"


def write_markdown(rows, path):
    run_columns = ["Method", "D", "C", "ρ", "τ", "Tmax", "adaptive"]
    sections = []
    for set_name in SET_ORDER:
        for group, _ in GROUPS:
            selected = [row for row in rows if row["set"] == set_name and row["group"] == group]
            if not selected:
                continue
            runs = [[row["method"], row["distractors"] or row["D"], row["C"], row["rho"],
                     row["tau"], row["T"], row["adaptive"]] for row in selected]
            # before / after reranking / ranking by Lowe matches only
            retrieval = markdown_table(
                run_columns + ["Q", "R@5 before", "R@10 before", "mAP before",
                               "R@5 after", "R@10 after", "mAP after",
                               "R@5 matches", "R@10 matches", "mAP matches"],
                [run + [row["queries"],
                        row["MeanBeforeRecall@5"], row["MeanBeforeRecall@10"],
                        row["MeanBeforeMAP@10"], row["MeanAfterRecall@5"],
                        row["MeanAfterRecall@10"], row["MeanAfterMAP@10"],
                        row["MeanMatchCountRecall@5"], row["MeanMatchCountRecall@10"],
                        row["MeanMatchCountMAP@10"]]
                 for run, row in zip(runs, selected)])
            # homography accuracy over 5 |Q| pairs, times in ms per query
            accuracy = markdown_table(
                run_columns + ["valid", "acc 1px", "acc 3px", "acc 5px", "corner px",
                               "T_cand ms", "T_verify ms", "T_total ms", "< C"],
                [run + [row["ValidHomographyRate"], row["AccurateHomographyRate1px"],
                        row["AccurateHomographyRate3px"], row["AccurateHomographyRate5px"],
                        row["MeanCornerErrorPx"],
                        milliseconds(row["AverageCandidateTimeMs"]),
                        milliseconds(row["AverageVerificationTimeMs"]),
                        milliseconds(row["AverageTotalTimeMs"]), row["QueriesBelowC"]]
                 for run, row in zip(runs, selected)])
            sections.append(f"## {set_name}, {group} queries\n\n### Retrieval\n\n{retrieval}\n"
                            f"### Homography accuracy and time\n\n{accuracy}")
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Verification results\n\n"
                   "D is the number of MIRFlickr distractors actually used.\n\n")
        file.write("\n".join(sections))


def main():
    if len(sys.argv) > 2:
        print(f"usage: {sys.argv[0]} [results_dir]", file=sys.stderr)
        return 1
    results_dir = Path(sys.argv[1] if len(sys.argv) == 2 else "results/verify")
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 1
    rows = collect(results_dir)
    if not rows:
        print(f"error: no finished runs in {results_dir}", file=sys.stderr)
        return 1
    write_csv(rows, results_dir / "summary.csv")
    write_markdown(rows, results_dir / "summary.md")
    print(f"collected {len(rows)} rows into {results_dir}/summary.csv and summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
