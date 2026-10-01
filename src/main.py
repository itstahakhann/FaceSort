"""Face Grouping & Auto-Organizer — command line (SPEC §4, §7, M3).

Pipeline: scan -> detect -> embed -> cluster -> auto-label known people ->
prompt for the rest -> sort.

Usage::

    python -m src.main
    python -m src.main --input ./photos --output ./sorted --mode copy
    python -m src.main --tolerance 0.45 --min-faces 3 -v
    python -m src.main --no-db         # ignore the persistent name DB
    python -m src.main --fetch-models  # one-time, only network call

Every ``--flag`` overrides the matching key from ``config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .clusterer import FaceCluster, FaceClusterer, FaceRecord, make_face_records
from .detector import process_images, resolve_workers
from .face_model import (
    ModelsNotInstalledError,
    download_models,
    model_dir,
    require_models,
)
from .image_loader import scan_images
from .names_db import NamesDB, open_names_db
from .organizer import PhotoOrganizer, normalize_person_name
from .ui.preview import PreviewBoard

logger = logging.getLogger("facesort")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

#: config keys that a CLI flag may override
OVERRIDE_KEYS = (
    "input_folder",
    "output_folder",
    "tolerance",
    "min_faces_per_cluster",
    "mode",
    "unknown_folder",
    "workers",
)

PROMPT = "Who is this? (name/skip) "


@dataclass
class PipelineStats:
    """Counters for the final report (SPEC F10)."""

    records: List[FaceRecord] = field(default_factory=list)
    images_scanned: int = 0
    images_unreadable: int = 0
    images_failed: int = 0
    images_without_faces: int = 0
    faces_detected: int = 0
    faces_embedded: int = 0
    seconds: float = 0.0

    @property
    def faces_in_clusters(self) -> int:
        return len(self.records)


def load_config(path: Path) -> Dict:
    """Read ``config.yaml`` (a missing file is a usage error, not a crash)."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - missing dependency
        raise RuntimeError(
            "PyYAML is not installed. Run: pip install -r requirements.txt"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping of keys to values")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facesort",
        description="Scan a folder of photos, group the faces of the same "
                    "person together, ask who each group is, and sort the "
                    "photos into named folders. Runs fully offline.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to config.yaml (default: %(default)s)")
    parser.add_argument("--input", dest="input_folder", metavar="FOLDER",
                        help="override input_folder: source of images")
    parser.add_argument("--output", dest="output_folder", metavar="FOLDER",
                        help="override output_folder: destination")
    parser.add_argument("--tolerance", type=float, metavar="EPS",
                        help="override tolerance: DBSCAN eps, cosine distance")
    parser.add_argument("--min-faces", dest="min_faces_per_cluster",
                        type=int, metavar="N",
                        help="override min_faces_per_cluster")
    parser.add_argument("--mode", choices=["copy", "move"],
                        help="override mode: copy or move the photos")
    parser.add_argument("--unknown-folder", dest="unknown_folder",
                        metavar="NAME",
                        help="override unknown_folder for skipped clusters")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="override workers: worker processes for face "
                             "detection (0 = auto up to the CPU count, "
                             "1 = single-process)")
    gui_or_cli = parser.add_mutually_exclusive_group()
    gui_or_cli.add_argument("--gui", action="store_true",
                            help="launch the Streamlit web UI instead of "
                                 "the interactive CLI")
    gui_or_cli.add_argument("--cli", action="store_true",
                            help="force the interactive CLI (the standalone "
                                 "executable opens the GUI on a bare "
                                 "double-click)")
    parser.add_argument("--no-db", action="store_true",
                        help="do not use the persistent name database for "
                             "this run (no auto-labelling, nothing saved)")
    parser.add_argument("--no-preview", action="store_true",
                        help="do not render/open the per-cluster thumbnail "
                             "montage (M2 preview)")
    parser.add_argument("--fetch-models", action="store_true",
                        help="download the buffalo_l model weights once "
                             "(the only step that needs the network), then exit")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for debug logging, -vv for very verbose")
    return parser


def apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    """Copy any flag the user passed on top of the config file values."""
    for key in OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def run_pipeline(config: Dict, progress_cb=None) -> PipelineStats:
    """Scan, detect and embed every photo (SPEC F1–F3). Never crashes on a
    single bad file: those are counted and logged.

    Large scans run the per-photo vision stage on a pool of worker
    processes (SPEC M6 — ``workers`` config: 0 = auto); smaller scans
    stay in one process.  ``progress_cb``, when given, is called as
    ``cb(done, total, path)`` after every photo — the GUI uses it to
    drive its progress bar.
    """
    stats = PipelineStats()

    paths = list(scan_images(config["input_folder"]))
    total = len(paths)
    logger.info("Found %d image(s) in %s", total, config["input_folder"])
    if not total:
        return stats

    # Fail fast with one clear message when the buffalo_l weights were
    # never fetched (the CLI and the GUI both catch this error); doing it
    # per photo would just print one warning per file.
    require_models()

    workers = resolve_workers(config.get("workers"))
    started = time.perf_counter()
    for done, result in enumerate(process_images(paths, workers), 1):
        stats.images_scanned += 1
        if done == 1 or done % 25 == 0:
            logger.info("Processing %d/%d ...", done, total)

        if result.status == "unreadable":
            stats.images_unreadable += 1
        else:
            if result.status == "failed":
                stats.images_failed += 1
            stats.faces_detected += result.faces_detected
            if not result.faces_detected:
                stats.images_without_faces += 1
            stats.faces_embedded += len(result.embeddings)
            stats.records.extend(
                make_face_records(result.path, result.embeddings)
            )

        if progress_cb is not None:
            try:
                progress_cb(done, total, result.path)
            except Exception as exc:  # a broken widget must not stop a scan
                logger.debug("progress callback failed: %s", exc)

    stats.seconds = time.perf_counter() - started
    return stats


COMMANDS = frozenset({"merge", "split", "show", "list", "rename", "help", "abort"})

COMMAND_HELP = """\
Commands (type these instead of a name):
  merge <cluster-id>          merge this cluster with another one
  split <n,n,...>             move the listed faces (the numbers above)
                              into a new cluster, reviewed next
  split <cluster-id> <n,n,...> split that other cluster the same way
  show <cluster-id>           show that cluster again (photos + preview)
  list                        list every cluster and its state
  rename <cluster-id> <name>  name another cluster without prompting
  help                        show this help
  abort                       quit without changing any files
Press Enter on its own, or type 'skip', to send the cluster to the
unknown folder. (So 'help', 'list', 'merge', 'split', 'show', 'rename'
and 'abort' cannot be used as person names.)"""


def _parse_command(answer: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return ``(verb, args)`` when the answer is a review command."""
    text = str(answer or "").strip()
    if not text:
        return None
    head, _, rest = text.partition(" ")
    if head.lower() in COMMANDS:
        return head.lower(), rest.strip()
    return None


def _parse_int(token: Optional[str]) -> Optional[int]:
    try:
        return int(str(token).strip())
    except (TypeError, ValueError):
        return None


def _parse_index_list(text: str) -> Optional[List[int]]:
    """Parse ``1,3 5`` into sorted unique 1-based indices (or ``None``)."""
    tokens = [token for token in text.replace(",", " ").split() if token]
    if not tokens:
        return None
    numbers: List[int] = []
    for token in tokens:
        value = _parse_int(token)
        if value is None or value < 1:
            return None
        numbers.append(value)
    return sorted(set(numbers))


def display_cluster(
    cluster: FaceCluster,
    position: int,
    total: int,
    label: Optional[str] = None,
    board=None,
    open_viewer: bool = False,
    numbered: bool = True,
) -> None:
    """Print one cluster: header, montage preview, and (optionally) its
    faces with the 1-based numbers the ``split`` command expects."""
    state = f" ({label})" if label else ""
    print(
        f"\n[{position}/{total}] Cluster #{cluster.cluster_id}{state}: "
        f"{cluster.size} face(s) in {len(cluster.image_paths)} photo(s)"
    )

    if board is not None:
        result = board.show(cluster, open_viewer=open_viewer)
        if result is not None:
            suffix = " (opened in the image viewer)" if result.opened else ""
            print(f"    preview: {result.path}{suffix}")

    if numbered:
        for index, face in enumerate(cluster.faces, 1):
            print(f"    ({index}) {face.image_path}")


def print_cluster_listing(
    clusters: Sequence[FaceCluster],
    auto_labels: Dict[int, str],
    names: Dict[int, Optional[str]],
) -> None:
    """One line per cluster for the ``list`` command."""
    print("\nClusters:")
    for cluster in clusters:
        cluster_id = cluster.cluster_id
        if cluster_id in auto_labels:
            state = f"auto: {auto_labels[cluster_id]}"
        elif cluster_id in names:
            state = f"named: {names[cluster_id]}" if names[cluster_id] else "skipped"
        else:
            state = "waiting for a name"
        print(f"  #{cluster_id}  {cluster.size} face(s) in "
              f"{len(cluster.image_paths)} photo(s)  [{state}]")


def review_clusters(
    clusters: List[FaceCluster],
    auto_labels: Optional[Dict[int, str]] = None,
    board=None,
    input_func=input,
    on_named=None,
) -> Tuple[Optional[Dict[int, Optional[str]]], Dict[str, int]]:
    """Walk every cluster: show its preview, accept a name or a command.

    This is the M2/M3 review loop (SPEC F5, F6, §10, §12):

    * auto-labelled clusters are displayed but never prompted;
    * every other cluster asks ``Who is this? (name/skip)``, and the answer
      may instead be a :data:`COMMANDS` command (``merge``, ``split``,
      ``show``, ``list``, ``rename``, ``help``, ``abort``);
    * ``clusters`` is edited *in place* — merges combine face lists, splits
      cut a new cluster out — so callers organise the corrected grouping.

    Returns ``(names, stats)`` where ``names`` maps cluster id to the name
    (``None`` = unknown folder) and ``stats`` counts what happened during
    review.  ``names is None`` means the user aborted; ``names[cid] is
    None`` after EOF means the remaining clusters were skipped.
    """
    auto_labels = dict(auto_labels or {})
    names: Dict[int, Optional[str]] = {}
    stats = {"prompted": 0, "auto": 0, "named": 0, "skipped": 0,
             "merged": 0, "split": 0}
    if not clusters:
        return names, stats

    next_cluster_id = max((c.cluster_id for c in clusters), default=-1) + 1
    queue = [cluster.cluster_id for cluster in clusters]
    dirty = True

    def find(cluster_id: int) -> Optional[FaceCluster]:
        return next((c for c in clusters if c.cluster_id == cluster_id), None)

    def positions(cluster: FaceCluster) -> Tuple[int, int]:
        for index, candidate in enumerate(clusters):
            if candidate.cluster_id == cluster.cluster_id:
                return index + 1, len(clusters)
        return 0, len(clusters)

    def label_for(cluster_id: int) -> Optional[str]:
        if cluster_id in auto_labels:
            return f"auto-labeled '{auto_labels[cluster_id]}'"
        if cluster_id in names:
            name = names[cluster_id]
            return f"named '{name}'" if name else "skipped"
        return None

    def handle_command(verb: str, args: str,
                       current: FaceCluster) -> str:
        """Run one review command; returns 'continue' | 'redisplay' | 'done'
        | 'abort'."""
        nonlocal next_cluster_id, dirty

        if verb == "help":
            print(COMMAND_HELP)
            return "continue"

        if verb == "list":
            print_cluster_listing(clusters, auto_labels, names)
            return "continue"

        if verb == "abort":
            return "abort"

        if verb == "show":
            target_id = _parse_int(args.split()[0]) if args.split() else None
            if target_id is None:
                print("usage: show <cluster-id>")
                return "continue"
            target = find(target_id)
            if target is None:
                print(f"No cluster #{target_id}.")
                return "continue"
            position, total = positions(target)
            display_cluster(target, position, total, label=label_for(target_id),
                            board=board, open_viewer=True)
            return "continue"

        if verb == "merge":
            tokens = args.split()
            other_id = _parse_int(tokens[0]) if tokens else None
            if other_id is None:
                print("usage: merge <cluster-id>   ('list' shows the ids)")
                return "continue"
            other = find(other_id)
            if other is None:
                print(f"No cluster #{other_id}.")
                return "continue"
            if other_id == current.cluster_id:
                print("A cluster cannot be merged with itself.")
                return "continue"

            # whatever name the other cluster carried is inherited
            inherited = names.pop(other_id, None)
            if inherited is None:
                inherited = auto_labels.pop(other_id, None)

            clusters.remove(other)
            if other_id in queue:
                queue.remove(other_id)
            for face in other.faces:
                face.cluster_id = current.cluster_id
            current.faces = list(current.faces) + list(other.faces)
            stats["merged"] += 1

            if inherited is not None:
                names[current.cluster_id] = inherited
                print(f"    Merged cluster #{other_id} into "
                      f"#{current.cluster_id} ({current.size} face(s)); "
                      f"keeping name '{inherited}'.")
                return "done"
            print(f"    Merged cluster #{other_id} into #{current.cluster_id} "
                  f"({current.size} face(s)).")
            return "redisplay"

        if verb == "split":
            tokens = args.split()
            if not tokens:
                print("usage: split <n,n,...>  or  split <cluster-id> <n,n,...>")
                return "continue"
            target = current
            index_tokens = tokens
            if len(tokens) >= 2:
                target_id = _parse_int(tokens[0])
                if target_id is not None:
                    found = find(target_id)
                    if found is None:
                        print(f"No cluster #{target_id}.")
                        return "continue"
                    target = found
                    index_tokens = tokens[1:]

            selected = _parse_index_list(" ".join(index_tokens))
            if selected is None:
                print("usage: split <face numbers, e.g. 1,3>")
                return "continue"
            face_count = len(target.faces)
            if selected[-1] > face_count:
                print(f"Cluster #{target.cluster_id} shows {face_count} "
                      f"face(s); pick numbers from 1..{face_count}.")
                return "continue"
            if len(selected) >= face_count:
                print("That would move every face — leave at least one "
                      "behind (or use merge instead).")
                return "continue"

            wanted = set(selected)
            moved = [face for i, face in enumerate(target.faces, 1)
                     if i in wanted]
            kept = [face for i, face in enumerate(target.faces, 1)
                    if i not in wanted]
            new_cluster = FaceCluster(cluster_id=next_cluster_id, faces=moved)
            for face in moved:
                face.cluster_id = next_cluster_id
            next_cluster_id += 1
            target.faces = kept
            clusters.append(new_cluster)
            stats["split"] += 1

            # review the new cluster next when we just split the current
            # one; otherwise keep it behind its target (or at the end)
            if target.cluster_id == current.cluster_id:
                queue.insert(index + 1, new_cluster.cluster_id)
            elif target.cluster_id in queue and queue.index(target.cluster_id) > index:
                queue.insert(queue.index(target.cluster_id) + 1,
                             new_cluster.cluster_id)
            else:
                queue.append(new_cluster.cluster_id)

            print(f"    Split off Cluster #{new_cluster.cluster_id} with "
                  f"{len(moved)} face(s).")
            if target.cluster_id == current.cluster_id:
                print(f"    Cluster #{current.cluster_id} keeps "
                      f"{len(kept)} face(s).")
                return "redisplay"
            print(f"    Cluster #{target.cluster_id} keeps {len(kept)} "
                  f"face(s); the new cluster is reviewed later.")
            return "continue"

        if verb == "rename":
            parts = args.split(None, 1)
            if len(parts) < 2 or _parse_int(parts[0]) is None:
                print("usage: rename <cluster-id> <name>")
                return "continue"
            target_id = _parse_int(parts[0])
            target = find(target_id)
            if target is None:
                print(f"No cluster #{target_id}.")
                return "continue"
            name = normalize_person_name(parts[1])
            auto_labels.pop(target_id, None)
            names[target_id] = name
            if name and on_named is not None:
                on_named(target, name)
            stats["named" if name else "skipped"] += 1
            if target_id == current.cluster_id:
                print(f"    -> {name if name else '(unknown folder)'}")
                stats["prompted"] += 1
                return "done"
            print(f"    Cluster #{target_id} -> "
                  f"{name if name else '(unknown folder)'}")
            return "continue"

        return "continue"  # pragma: no cover - COMMANDS guards this

    index = 0
    while index < len(queue):
        cluster = find(queue[index])
        if cluster is None:  # merged away while reviewing an earlier one
            index += 1
            continue
        cluster_id = cluster.cluster_id

        # clusters already answered for (auto-labeled or renamed ahead)
        # are shown for review but never prompted
        if cluster_id in auto_labels or cluster_id in names:
            position, total = positions(cluster)
            display_cluster(cluster, position, total,
                            label=label_for(cluster_id),
                            board=board, open_viewer=False, numbered=False)
            names.setdefault(cluster_id, auto_labels.get(cluster_id))
            if cluster_id in auto_labels:
                stats["auto"] += 1
            index += 1
            continue

        dirty = True
        while True:
            if dirty:
                position, total = positions(cluster)
                display_cluster(cluster, position, total, board=board,
                                open_viewer=True)
                dirty = False

            try:
                raw = input_func(PROMPT)
            except EOFError:
                print("\nNo input available — skipping the remaining "
                      "clusters.")
                for remaining_id in queue[index:]:
                    remaining = find(remaining_id)
                    if remaining is None:
                        continue
                    if remaining_id in auto_labels:
                        names[remaining_id] = auto_labels[remaining_id]
                        stats["auto"] += 1
                    elif remaining_id not in names:
                        names[remaining_id] = None
                return names, stats

            command = _parse_command(raw)
            if command is not None:
                outcome = handle_command(command[0], command[1], cluster)
                if outcome == "abort":
                    return None, stats
                if outcome == "done":
                    stats["prompted"] += 1
                    break
                if outcome == "redisplay":
                    dirty = True
                continue

            name = normalize_person_name(raw)
            names[cluster_id] = name
            print(f"    -> {name if name else '(unknown folder)'}")
            if name and on_named is not None:
                on_named(cluster, name)
            stats["prompted"] += 1
            stats["named" if name else "skipped"] += 1
            break

        index += 1

    return names, stats


def prompt_cluster_names(
    clusters: Sequence[FaceCluster],
    input_func=input,
    on_named=None,
) -> Dict[int, Optional[str]]:
    """Backwards-compatible front-end for :func:`review_clusters`.

    Shows every cluster and asks ``Who is this? (name/skip)`` (SPEC F5/F6);
    ``None`` means skip/unknown.  Closed stdin skips the remaining clusters,
    and ``on_named(cluster, name)`` fires right after every named answer so
    it is persisted even if the run is interrupted later.  Raises
    ``KeyboardInterrupt`` if the user types ``abort``.
    """
    names, _stats = review_clusters(list(clusters), auto_labels={},
                                    board=None, input_func=input_func,
                                    on_named=on_named)
    if names is None:
        raise KeyboardInterrupt
    return names


def auto_label_clusters(
    clusters: Sequence[FaceCluster],
    db: Optional[NamesDB],
    tolerance: float,
) -> Tuple[Dict[int, str], List[FaceCluster]]:
    """Split clusters into DB matches and clusters that still need a name.

    A cluster whose centroid is within ``tolerance`` (cosine distance, the
    same metric as clustering) of a known person is auto-labelled — the
    prompt is skipped for it (M3 / SPEC F11).  Everything else is returned
    untouched so the user is asked only about genuinely unknown people.
    The per-cluster display happens later, in :func:`review_clusters`, so
    the preview montage appears exactly once per cluster.
    """
    auto_labels: Dict[int, str] = {}
    pending: List[FaceCluster] = []

    for cluster in clusters:
        match = None
        if db is not None and cluster.centroid is not None:
            match = db.find_match(
                cluster.centroid, tolerance,
                eye_embedding_centroid=cluster.eye_centroid,
            )
        if match is None:
            pending.append(cluster)
            continue
        auto_labels[cluster.cluster_id] = match.person_name
        logger.debug("Cluster #%s auto-labeled as %r (distance %.3f, %s).",
                     cluster.cluster_id, match.person_name, match.distance,
                     "eye region + face" if match.used_eye else "face only")

    if auto_labels:
        print(f"\n{len(auto_labels)} cluster(s) matched the name database "
              f"and need no answer.")
    return auto_labels, pending


def label_and_fuse(
    clusters: Sequence[FaceCluster],
    db: Optional[NamesDB],
    tolerance: float,
) -> Tuple[Dict[int, str], List[FaceCluster]]:
    """Auto-label clusters against the database, then fuse remembered merges.

    This is the two-stage pass the desktop app and the CLI both want:
    :func:`auto_label_clusters` decides *who* each group is, and the fuse step
    then collapses groups that the database already knows are one person — so
    a merge the user confirmed in one run survives every later run even when
    the fused distance did not group them on its own.
    """
    labels, pending = auto_label_clusters(clusters, db, tolerance)
    # The fused list is the list of record objects; renumbering in place
    # afterwards is what keeps FaceRecord.cluster_id consistent for the
    # caller.  ``merge_clusters_by_memory`` returns new FaceCluster objects
    # only when a fuse actually happened.
    fused = merge_clusters_by_memory(clusters, labels)
    if len(fused) == len(clusters):
        return labels, pending
    fused = renumber_clusters(fused)

    print(f"Merged {len(clusters) - len(fused)} group(s) the name database "
          f"already knows are the same person.")
    # Re-derive the labels for the new ids: a fused group takes the name of
    # the (auto-labelled) cluster it was built from.
    by_face: Dict[int, str] = {}
    for cluster in clusters:
        name = labels.get(cluster.cluster_id)
        for face in cluster.faces:
            by_face[id(face)] = name
    relabelled: Dict[int, str] = {}
    for cluster in fused:
        for face in cluster.faces:
            name = by_face.get(id(face))
            if name:
                relabelled[cluster.cluster_id] = name
                break
    pending = [c for c in fused if not relabelled.get(c.cluster_id)]
    return relabelled, pending


def merge_clusters_by_memory(
    clusters: Sequence[FaceCluster],
    auto_labels: Mapping[int, str],
) -> List[FaceCluster]:
    """Fuse clusters the name database says are the same person.

    Remembering a merge is not enough on its own: a re-scan re-clusters from
    scratch, so a childhood group and an adult group of one person come back
    as two clusters again.  When both already carry the *same* auto-label
    they are the same person by definition, so they are fused here.

    This is what makes the manual merge stick across runs: the clusterer's
    fused distance usually gets it right, and this catches the cases it does
    not (very blurred childhood photos, a very wide age gap).

    Clusters keep the first member's order and id, so names and thumbnails
    stay stable.  Returns a new list; the input is not modified.
    """
    by_name: Dict[str, List[FaceCluster]] = {}
    for cluster in clusters:
        name = auto_labels.get(cluster.cluster_id)
        if name:
            by_name.setdefault(name, []).append(cluster)

    merged: List[FaceCluster] = []
    consumed: set = set()
    leader_of: Dict[int, FaceCluster] = {}   # source cluster id -> fused group
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        fused = FaceCluster(cluster_id=group[0].cluster_id)
        for cluster in group:
            fused.faces.extend(cluster.faces)
            consumed.add(id(cluster))
            leader_of[cluster.cluster_id] = fused
        for face in fused.faces:
            face.cluster_id = fused.cluster_id
        logger.info(
            "Fused %d group(s) remembered as %r into one (%d faces).",
            len(group), name, fused.size,
        )
        merged.append(fused)

    # Preserve the original ordering: a fused group appears where its first
    # member was, and is emitted only once.
    ordered: List[FaceCluster] = []
    emitted: set = set()
    for cluster in clusters:
        fused = leader_of.get(cluster.cluster_id)
        if fused is not None:
            if fused.cluster_id not in emitted:
                emitted.add(fused.cluster_id)
                ordered.append(fused)
        else:
            ordered.append(cluster)
    for fused in merged:
        if fused.cluster_id not in emitted:
            emitted.add(fused.cluster_id)
            ordered.append(fused)
    return ordered


def renumber_clusters(clusters: Sequence[FaceCluster]) -> List[FaceCluster]:
    """Reassign contiguous ids 0..n-1, in order, and return the list."""
    result = list(clusters)
    for index, cluster in enumerate(result):
        cluster.cluster_id = index
        for face in cluster.faces:
            face.cluster_id = index
    return result


def save_named_cluster(db: Optional[NamesDB], cluster: FaceCluster,
                       name: str) -> bool:
    """Persist one freshly named cluster (M3). Never raises: a failing DB
    must not lose the naming the user just did.

    The periocular centroid is stored alongside the whole-face one when the
    cluster has it, so later runs can match this person across the age gap.
    """
    if db is None or cluster.centroid is None:
        return False
    try:
        db.upsert_person(
            name,
            cluster.centroid,
            cluster.image_paths,
            faces_seen=cluster.size,
            eye_embedding_centroid=cluster.eye_centroid,
            eye_faces_seen=cluster.size if cluster.eye_centroid is not None else 0,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - report, don't abort
        logger.warning("Could not remember %r in the name DB: %s", name, exc)
        return False


def print_report(
    stats: PipelineStats,
    clusters: Sequence[FaceCluster],
    summary,
    db: Optional[NamesDB] = None,
    review_stats: Optional[Dict[str, int]] = None,
    remembered: int = 0,
) -> None:
    """Print the end-of-run summary (SPEC F10 / §4.9)."""
    rate = stats.images_scanned / stats.seconds if stats.seconds > 0 else 0.0
    faces_named = sum(cluster.size for cluster in clusters)

    print("\n" + "=" * 46)
    print(" Summary")
    print("=" * 46)
    print(f"Photos scanned       : {stats.images_scanned}")
    print(f"  unreadable         : {stats.images_unreadable}")
    print(f"  failed to process  : {stats.images_failed}")
    print(f"  no faces found     : {stats.images_without_faces}")
    print(f"Faces detected       : {stats.faces_detected}")
    print(f"Faces embedded       : {stats.faces_embedded}")
    print(f"Clusters formed      : {len(clusters)} "
          f"({faces_named} face(s) kept, "
          f"{stats.faces_embedded - faces_named} in smaller clusters)")
    print(f"Processed in         : {stats.seconds:.1f}s ({rate:.1f} img/s)")
    if review_stats is not None:
        print("-" * 46)
        print(f"Auto-labeled         : {review_stats.get('auto', 0)} cluster(s)")
        print(f"Prompted for         : {review_stats.get('prompted', 0)} cluster(s)")
        print(f"Clusters merged      : {review_stats.get('merged', 0)}")
        print(f"Clusters split       : {review_stats.get('split', 0)}")
        print(f"Remembered new names : {remembered}")
        if db is not None:
            print(f"Name DB              : {db.path} "
                  f"({db.person_count()} known person(s))")
    print("-" * 46)
    print(summary.format())
    print("=" * 46)


def _app_script_path() -> Path:
    """Filesystem path of the Streamlit app script (SPEC M4).

    Source checkouts run it straight from ``src/ui/app.py``; a PyInstaller
    build ships it as a data file next to the bundled modules
    (``ui/app.py`` under ``sys._MEIPASS``).
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS",
                            str(Path(__file__).resolve().parent)))
        return base / "ui" / "app.py"
    return Path(__file__).resolve().parent / "ui" / "app.py"


def run_gui(extra_args: Optional[Sequence[str]] = None) -> int:
    """Launch the Streamlit web UI in this process (SPEC M4 / M5).

    Used by ``--gui`` and by the standalone executable on a bare
    double-click.  It goes through Streamlit's public CLI so configuration
    precedence (flags, ``STREAMLIT_*`` env vars, config files) behaves
    exactly like ``streamlit run src/ui/app.py`` — and blocks until the
    server stops.
    """
    # Offline guarantee (SPEC §6): never let Streamlit phone home, even in a
    # frozen exe where the repo's .streamlit/config.toml is not around.
    # setdefault keeps any explicit user configuration authoritative.
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    if getattr(sys, "frozen", False):
        # PyInstaller extracts streamlit outside site-packages, which trips
        # Streamlit's "am I a source checkout?" heuristic (it checks for
        # "site-packages" in config.py's path) and silently enables
        # development mode — that forbids setting server.port (hard crash
        # on _check_conflicts) and turns on debug logging. Force the
        # production behaviour unless the user configured otherwise.
        os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")

    from streamlit.web import cli as stcli  # heavy import, kept lazy

    # --server.showEmailPrompt=false: skip Streamlit's first-run email
    # onboarding (it would abort the exe on a non-interactive stdin, and it
    # exists only to personalise telemetry we disable above anyway).
    # Placed before extra_args so an explicit user flag still wins.
    sys.argv = ["streamlit", "run", str(_app_script_path()),
                "--server.showEmailPrompt=false",
                *(extra_args or ())]
    try:
        stcli.main()
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except KeyboardInterrupt:
        print("\nGUI stopped.", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)

    level = logging.WARNING if args.verbose == 0 else (
        logging.INFO if args.verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.fetch_models:
        try:
            path = download_models()
        except Exception as exc:
            print(f"Error downloading models: {exc}", file=sys.stderr)
            return 2
        print(f"Model ready: {path}")
        print(f"Location:    {model_dir()}")
        print("Inference from now on runs fully offline.")
        return 0

    # Legacy: the Streamlit UI is still reachable, but the shipped desktop
    # app is Electron (electron/main.js spawns src/api/server.py, not this
    # module), so no frozen build routes here any more.
    if args.gui:
        return run_gui()

    try:
        config = apply_overrides(load_config(args.config), args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    backend = str(config.get("detector_backend", "retinaface")).lower()
    if backend not in ("retinaface", "buffalo_l", "insightface"):
        logger.warning(
            "detector_backend=%r is not implemented yet; using insightface "
            "retinaface (buffalo_l).", backend,
        )

    try:
        stats = run_pipeline(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ModelsNotInstalledError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAborted — no files were changed.", file=sys.stderr)
        return 130

    clusters = FaceClusterer.from_config(config).cluster(stats.records)
    try:
        organizer = PhotoOrganizer.from_config(config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    db = None if args.no_db else open_names_db(config)
    try:
        if not clusters:
            print("\nNo clusters to name — nothing was moved or copied.")
            print_report(stats, clusters, organizer.organize([]), db=db)
            return 0

        try:
            tolerance = float(config.get("tolerance", 0.5))
        except (TypeError, ValueError):
            logger.warning("tolerance=%r is not a number; using 0.5",
                           config.get("tolerance"))
            tolerance = 0.5

        # M3: known people are labelled automatically, so the prompt only ever asks
        # about clusters the DB could not recognise.  Groups the database
        # already knows are one person are fused here, which is how a merge
        # confirmed in an earlier run (or in the desktop app) survives.
        auto_labels, _pending = label_and_fuse(clusters, db, tolerance)

        remembered = 0

        def remember(cluster: FaceCluster, name: str) -> None:
            nonlocal remembered
            if save_named_cluster(db, cluster, name):
                remembered += 1

        # M2 + §10: show a thumbnail montage before each question, and let
        # the user merge/split clusters while reviewing.
        board = PreviewBoard(enabled=not args.no_preview)
        try:
            names, review_stats = review_clusters(
                clusters, auto_labels=auto_labels, board=board,
                on_named=remember,
            )
        except KeyboardInterrupt:
            print("\nAborted — no photos were changed (any names you "
                  "already typed were saved for the next run).",
                  file=sys.stderr)
            return 130
        finally:
            board.cleanup()

        if names is None:  # the user typed 'abort'
            print("\nAborted — no photos were changed (any names you "
                  "already typed were saved for the next run).",
                  file=sys.stderr)
            return 130

        named_clusters = [(names.get(cluster.cluster_id), cluster)
                          for cluster in clusters]
        try:
            summary = organizer.organize(named_clusters)
        except (ValueError, OSError) as exc:
            print(f"Error organising photos: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nAborted — some photos may already have been copied.",
                  file=sys.stderr)
            return 130

        print()
        print_report(stats, clusters, summary, db=db,
                     review_stats=review_stats, remembered=remembered)
        return 0 if summary.ok else 1
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    # No-op outside a multiprocessing child / frozen build; in a frozen
    # executable, pool workers re-launch the exe and must exit here instead
    # of starting the app again (SPEC M5 + M6).
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
