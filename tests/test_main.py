"""Unit tests for the M1 CLI (SPEC §4, F6, and the config-override flags).

Run with::

    python tests/test_main.py
    pytest tests/
"""

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.clusterer import FaceCluster, FaceRecord
from src.main import PROMPT, apply_overrides, build_parser, load_config, prompt_cluster_names
from src.organizer import normalize_person_name

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


def cluster(cluster_id, *names):
    faces = [
        FaceRecord(image_path=Path(name), bbox=(0, 0, 5, 5),
                   embedding=np.ones(4, dtype=np.float32))
        for name in names
    ]
    for face in faces:          # the real clusterer stamps these too
        face.cluster_id = cluster_id
    return FaceCluster(cluster_id=cluster_id, faces=faces)


def test_cli_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--input", "D:/pics",
        "--output", "D:/sorted",
        "--tolerance", "0.42",
        "--mode", "move",
        "--min-faces", "4",
        "--unknown-folder", "misc",
        "-v",
    ])
    config = {
        "input_folder": "./input_photos",
        "output_folder": "./grouped_photos",
        "tolerance": 0.5,
        "min_faces_per_cluster": 2,
        "mode": "copy",
        "unknown_folder": "_unknown",
    }
    apply_overrides(config, args)
    check("--input overrides input_folder", config["input_folder"] == "D:/pics")
    check("--output overrides output_folder", config["output_folder"] == "D:/sorted")
    check("--tolerance overrides tolerance", config["tolerance"] == 0.42)
    check("--mode overrides mode", config["mode"] == "move")
    check("--min-faces overrides min_faces_per_cluster",
          config["min_faces_per_cluster"] == 4)
    check("--unknown-folder overrides unknown_folder",
          config["unknown_folder"] == "misc")

    # no flags -> config.yaml values untouched
    defaults = {"tolerance": 0.5, "mode": "copy"}
    apply_overrides(defaults, parser.parse_args([]))
    check("no flags leaves the config alone",
          defaults == {"tolerance": 0.5, "mode": "copy"})

    # --mode only accepts copy/move
    with contextlib.redirect_stderr(io.StringIO()):
        rejects_bad_mode = _bad_mode(parser)
    check("--mode rejects other values", rejects_bad_mode)
    check("--fetch-models is available",
          parser.parse_args(["--fetch-models"]).fetch_models)


def _bad_mode(parser):
    try:
        parser.parse_args(["--mode", "link"])
        return False
    except SystemExit:
        return True


def test_load_config():
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.yaml")
    check("config.yaml loads", config["input_folder"] == "./input_photos"
          and config["tolerance"] == 0.5)
    try:
        load_config(root / "no_such_config.yaml")
        check("missing config raises FileNotFoundError", False)
    except FileNotFoundError:
        check("missing config raises FileNotFoundError", True)


def test_prompt_flow():
    clusters = [
        cluster(0, "a1.jpg", "a2.jpg"),
        cluster(1, "b1.jpg"),
        cluster(2, "c1.jpg"),
    ]
    answers = iter(["Alice", "skip", "Bob"])  # raw user input
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(answers)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        names = prompt_cluster_names(clusters, input_func=fake_input)

    check("asks once per cluster", len(prompts) == 3, str(prompts))
    check("prompt text matches the spec wording",
          all(p == PROMPT for p in prompts) and PROMPT == "Who is this? (name/skip) ",
          repr(PROMPT))
    check("names captured per cluster",
          names == {0: "Alice", 1: None, 2: "Bob"}, str(names))
    text = out.getvalue()
    check("image paths are printed for each cluster",
          "a1.jpg" in text and "a2.jpg" in text and "b1.jpg" in text)
    check("cluster header shows counts", "Cluster #0" in text
          and "2 face(s)" in text)

    # closed stdin skips everything instead of crashing
    def eof_input(prompt):
        raise EOFError

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        names = prompt_cluster_names(clusters, input_func=eof_input)
    check("EOF skips all clusters gracefully",
          names == {0: None, 1: None, 2: None}, str(names))
    check("names are normalised through the organizer rules",
          normalize_person_name("  Alice  ") == "Alice")


def test_auto_label_flow():
    """M3: known people are shown but never prompted; only the rest are."""
    from src.main import (auto_label_clusters, review_clusters,
                          save_named_cluster)
    from src.names_db import NamesDB

    known = [cluster(0, "k1.jpg", "k2.jpg")]
    unknown = [cluster(1, "u1.jpg")]
    unknown[0].faces[0] = FaceRecord(image_path=Path("u1.jpg"),
                                     bbox=(0, 0, 5, 5),
                                     embedding=-np.ones(4, dtype=np.float32))

    with NamesDB(":memory:") as db:
        db.upsert_person("Alice", known[0].centroid, ["old.jpg"])

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            auto, pending = auto_label_clusters(known + unknown, db, 0.5)

        check("DB match is auto-labeled", auto == {0: "Alice"}, str(auto))
        check("unknown cluster still needs a prompt",
              [c.cluster_id for c in pending] == [1], str(pending))
        check("match summary printed", "1 cluster(s) matched" in out.getvalue())

        # full review: the auto cluster is *displayed*, only the unknown
        # cluster reaches a prompt — exactly one question asked
        answers = iter(["Carol"])
        prompts = []
        saved = []

        def fake_input(prompt):
            prompts.append(prompt)
            return next(answers)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            names, stats = review_clusters(
                known + unknown,
                auto_labels=auto,
                input_func=fake_input,
                # exactly how main.py wires it: record + persist immediately
                on_named=lambda c, n: (
                    saved.append((c.cluster_id, n)),
                    save_named_cluster(db, c, n),
                ),
            )
        text = out.getvalue()
        check("auto cluster is displayed with its label",
              "(auto-labeled 'Alice')" in text, text[:160])
        check("auto cluster shows no photo list",
              "k1.jpg" not in text, text[:160])
        check("only unknown clusters are prompted",
              names == {0: "Alice", 1: "Carol"} and len(prompts) == 1,
              f"{names} prompts={len(prompts)}")
        check("review stats count auto vs prompted",
              stats["auto"] == 1 and stats["prompted"] == 1, str(stats))
        check("on_named callback fires for the answer",
              saved == [(1, "Carol")], str(saved))
        check("typed name is persisted",
              db.get_person("Carol") is not None)

        # saving works through the helper too, and skips silently without a DB
        check("save_named_cluster persists",
              save_named_cluster(db, cluster(2, "n1.jpg"), "Dave")
              and db.get_person("Dave") is not None)
        check("save_named_cluster without a db is a no-op",
              not save_named_cluster(None, cluster(3, "x.jpg"), "Eve"))

        # a skipped answer must not touch the database
        before = db.person_count()
        saved.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            skipped, _ = review_clusters(
                [cluster(4, "s1.jpg")],
                input_func=lambda prompt: "skip",
                on_named=lambda c, n: saved.append(n),
            )
        check("skip answers map to None (unknown folder)",
              skipped == {4: None}, str(skipped))
        check("skipped names are never persisted",
              saved == [] and db.person_count() == before,
              f"saved={saved} count={db.person_count()}")


def _scripted_review(clusters, answers, auto_labels=None, on_named=None):
    """Run review_clusters with canned answers; capture output + prompts."""
    from src.main import review_clusters

    answers = iter(answers)
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(answers)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = review_clusters(clusters, auto_labels=auto_labels or {},
                                 input_func=fake_input, on_named=on_named)
    return result, prompts, out.getvalue()


def test_review_commands():
    """M2/§10: merge, split, show, list, rename, help, abort during review."""
    from src.main import COMMAND_HELP

    # --- split the current cluster in two, both halves get named ---
    clusters = [cluster(0, "a1.jpg", "a2.jpg"), cluster(1, "b1.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["split 2", "Alice", "Bob", "Carol"])
    check("split creates a new cluster",
          len(clusters) == 3 and clusters[0].size == 1
          and clusters[2].size == 1,
          f"n={len(clusters)} sizes={[c.size for c in clusters]}")
    check("split updates face ownership",
          clusters[2].faces[0].cluster_id == clusters[2].cluster_id
          and clusters[0].faces[0].cluster_id == 0,
          str([f.cluster_id for c in clusters for f in c.faces]))
    check("split message printed", "Split off Cluster #2" in text)
    check("split halves are both named",
          names == {0: "Alice", 2: "Bob", 1: "Carol"}, str(names))
    check("split counts in stats", stats["split"] == 1
          and stats["prompted"] == 3, str(stats))

    # --- merge the current cluster with a later, unnamed one ---
    clusters = [cluster(0, "a1.jpg", "a2.jpg"), cluster(1, "b1.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["merge 1", "Alice"])
    check("merge combines the face lists",
          len(clusters) == 1 and clusters[0].size == 3,
          f"n={len(clusters)} size={clusters[0].size if clusters else 0}")
    check("merge message printed", "Merged cluster #1 into #0" in text)
    check("merged cluster still asks for one name",
          names == {0: "Alice"} and len(prompts) == 2, str(names))
    check("merge counts in stats", stats["merged"] == 1, str(stats))

    # --- merge with an auto-labeled cluster: inherits its name, no prompt ---
    clusters = [cluster(0, "a1.jpg"), cluster(1, "b1.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["merge 1"], auto_labels={1: "Bob"})
    check("merge with a known cluster inherits its name",
          names == {0: "Bob"} and len(clusters) == 1, str(names))
    check("inherited name skips the question", len(prompts) == 1,
          str(len(prompts)))
    check("inherit message printed", "keeping name 'Bob'" in text)

    # --- merge with a cluster that was already named earlier ---
    clusters = [cluster(0, "a1.jpg"), cluster(1, "b1.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["Alice", "merge 0"])
    check("merge with an earlier answer keeps that name",
          names == {1: "Alice"} and len(clusters) == 1
          and clusters[0].size == 2, str(names))
    check("earlier cluster is absorbed", "Merged cluster #0" in text)

    # --- help / list / show ---
    clusters = [cluster(0, "a1.jpg"), cluster(1, "b1.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["help", "list", "show 1", "Bob", "Alice"])
    check("help prints the command list", "merge <cluster-id>" in text)
    check("list prints every cluster",
          "Clusters:" in text and "#0" in text and "#1" in text)
    check("show re-displays another cluster with its numbers",
          "(1) " in text and "b1.jpg" in text, text[:200])
    check("commands do not consume a name slot",
          names == {0: "Bob", 1: "Alice"} and len(prompts) == 5
          and stats["prompted"] == 2, str(names))

    # --- rename another cluster without prompting for it ---
    clusters = [cluster(0, "a1.jpg"), cluster(1, "b1.jpg"),
                cluster(2, "c1.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["rename 2 Zed", "Alice", "Bob"])
    check("rename ahead fills the name",
          names == {0: "Alice", 1: "Bob", 2: "Zed"}, str(names))
    check("renamed cluster is shown, not prompted",
          len(prompts) == 3 and "named 'Zed'" in text,
          f"prompts={len(prompts)}")
    check("rename counts in stats", stats["named"] == 3, str(stats))

    # --- split an auto-labeled (wrongly clustered) cluster from outside ---
    clusters = [cluster(0, "a1.jpg"), cluster(1, "b1.jpg", "b2.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["split 1 2", "Alice", "Carol"],
        auto_labels={1: "Bob"})
    check("auto-labeled cluster can be split (SPEC §10)",
          len(clusters) == 3 and clusters[1].size == 1
          and clusters[2].size == 1,
          f"n={len(clusters)} sizes={[c.size for c in clusters]}")
    check("auto label survives on the part that stayed",
          names.get(1) == "Bob", str(names))
    check("split-off half is prompted normally",
          names.get(0) == "Alice" and names.get(2) == "Carol", str(names))

    # --- invalid commands are explained, then asking continues ---
    clusters = [cluster(0, "a1.jpg", "a2.jpg")]
    (names, stats), prompts, text = _scripted_review(
        clusters, ["merge", "merge 99", "split", "split 9", "split 1,2",
                   "Alice"])
    check("usage errors printed",
          "usage: merge <cluster-id>" in text
          and "usage: split" in text, text[:300])
    check("unknown cluster id reported", "No cluster #99" in text)
    check("out-of-range face number reported",
          "shows 2 face(s)" in text, text[:300])
    check("moving every face is refused",
          "would move every face" in text)
    check("asking continues after bad commands",
          names == {0: "Alice"} and len(prompts) == 6, str(names))
    check("bad commands are not counted as answers",
          stats["prompted"] == 1, str(stats))

    # --- abort stops the whole review ---
    (names, _stats), _prompts, text = _scripted_review(
        [cluster(0, "a.jpg")], ["abort"])
    check("abort ends the review without names", names is None)


def test_no_db_flag():
    parser = build_parser()
    check("--no-db parses", parser.parse_args(["--no-db"]).no_db is True)
    check("--no-db defaults off", parser.parse_args([]).no_db is False)
    check("--no-preview parses",
          parser.parse_args(["--no-preview"]).no_preview is True)
    check("--no-preview defaults off",
          parser.parse_args([]).no_preview is False)


def main():
    test_cli_flags()
    test_load_config()
    test_prompt_flow()
    test_auto_label_flow()
    test_review_commands()
    test_no_db_flag()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All main/CLI tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
