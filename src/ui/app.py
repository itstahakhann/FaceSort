"""Streamlit front-end (M4, SPEC §11 / §4).

Run it with::

    streamlit run src/ui/app.py

Layout:

* **Sidebar** — the config.yaml settings: input/output folders, tolerance,
  copy-vs-move mode (plus a *Re-scan* button).
* **Main area** — a progress bar while the face pipeline runs, then one
  card per cluster showing its thumbnail montage and a
  ``Who is this? (name/skip)`` input pre-filled with the name the
  persistent DB recognised, and finally a **Run Organizer** button that
  copies/moves the photos.

Everything is local (SPEC §6): no network call is made at any point, and
scan results live in ``st.session_state`` so typing a name, moving a
slider, or any other interaction re-renders the page *without* re-running
the expensive pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import streamlit as st

#: Streamlit executes this file with an unclear package context, so make
#: the project root importable no matter how the app was started.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import main as cli  # noqa: E402  (monkeypatchable: src.main.*)
from src.clusterer import FaceCluster, FaceClusterer, FaceRecord  # noqa: E402
from src.face_model import ModelsNotInstalledError  # noqa: E402
from src.names_db import open_names_db  # noqa: E402
from src.organizer import PhotoOrganizer, normalize_person_name  # noqa: E402
from src.ui.preview import CELL_SIZE, build_montage  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "NAME_KEY_PREFIX",
    "auto_names_for",
    "collect_names",
    "current_config",
    "init_session",
    "main",
    "montage_for",
    "persist_names",
    "render_sidebar",
    "scan_signature",
]

NAME_KEY_PREFIX = "cluster_name_"
# A PyInstaller build extracts its data files under sys._MEIPASS, where the
# repo layout (and PROJECT_ROOT above) does not exist — use the bundled
# config there instead.
if getattr(sys, "frozen", False):
    CONFIG_PATH = Path(getattr(sys, "_MEIPASS", ".")) / "config.yaml"
else:
    CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ----------------------------------------------------------------- state
def init_session(defaults: Dict) -> None:
    """Seed the sidebar widget values from config.yaml exactly once.

    The values are written to ``st.session_state`` *before* the widgets
    are created (the documented Streamlit pattern), so the widgets own
    them from then on and every later interaction keeps its value.
    """
    pairs = (
        ("cfg_input", str(defaults.get("input_folder", "./input_photos"))),
        ("cfg_output", str(defaults.get("output_folder", "./grouped_photos"))),
        ("cfg_tolerance", float(defaults.get("tolerance", 0.5))),
        ("cfg_mode", str(defaults.get("mode", "copy"))),
        ("cfg_min_faces", int(defaults.get("min_faces_per_cluster", 2))),
        ("cfg_unknown", str(defaults.get("unknown_folder", "_unknown"))),
        ("cfg_db", str(defaults.get("names_db", ""))),
    )
    for key, value in pairs:
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state["cfg_mode"] not in ("copy", "move"):
        st.session_state["cfg_mode"] = "copy"


def current_config() -> Dict:
    """The live configuration assembled from the sidebar widgets."""
    return {
        "input_folder": st.session_state["cfg_input"],
        "output_folder": st.session_state["cfg_output"],
        "tolerance": float(st.session_state["cfg_tolerance"]),
        "min_faces_per_cluster": int(st.session_state["cfg_min_faces"]),
        "mode": st.session_state["cfg_mode"],
        "unknown_folder": st.session_state["cfg_unknown"],
        "names_db": st.session_state["cfg_db"],
    }


def render_sidebar() -> None:
    """Sidebar: configuration + cache control (SPEC §9 keys)."""
    st.sidebar.title("Configuration")
    st.sidebar.text_input("Input folder", key="cfg_input",
                          help="Folder that is scanned for photos")
    st.sidebar.text_input("Output folder", key="cfg_output",
                          help="Destination: one sub-folder per person")
    st.sidebar.number_input("Tolerance (cosine distance)", min_value=0.05,
                            max_value=1.00, step=0.05, key="cfg_tolerance",
                            help="Higher = looser clusters and looser "
                                 "name matching")
    st.sidebar.radio("Mode", options=["copy", "move"], key="cfg_mode",
                     horizontal=True,
                     help="copy keeps the originals, move relocates them")
    st.sidebar.divider()
    if st.sidebar.button("Re-scan photos", key="re_scan",
                         help="Throw away the cached scan and run the face "
                              "pipeline again"):
        clear_scan_state()
        st.rerun()
    st.sidebar.caption(
        "Runs fully offline — no photo, face, or name ever leaves this "
        "machine. Settings come from `config.yaml` and can be changed here."
    )


def clear_scan_state() -> None:
    """Drop everything that depends on the previous scan."""
    for key in ("records", "scan_stats", "cluster_signature"):
        if key in st.session_state:
            del st.session_state[key]
    clear_names_and_montages()


def clear_names_and_montages() -> None:
    """Remove per-cluster name inputs and cached montages."""
    doomed = [key for key in list(st.session_state.keys())
              if key.startswith(NAME_KEY_PREFIX)
              or key.startswith("montage_")]
    for key in doomed:
        del st.session_state[key]


def scan_signature(config: Dict) -> str:
    """Cheap identity of a scan: only the input folder matters (SPEC §6)."""
    return str(config["input_folder"])


def cached_scan_matches(config: Dict) -> bool:
    stats = st.session_state.get("scan_stats")
    return bool(stats) and stats.get("scan") == scan_signature(config)


# ------------------------------------------------------------------ scan
def run_scan(config: Dict) -> Dict:
    """Run the face pipeline with a live progress bar; cache the result."""
    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(done: int, total: int, path) -> None:
        progress.progress(min(1.0, done / max(1, total)))
        status.caption(f"Processing {done}/{total}: {Path(path).name}")

    try:
        stats = cli.run_pipeline(config, progress_cb=on_progress)
    finally:
        progress.empty()
        status.empty()

    payload = {
        "scan": scan_signature(config),
        "records": stats.records,
        "images_scanned": stats.images_scanned,
        "images_unreadable": stats.images_unreadable,
        "images_failed": stats.images_failed,
        "images_without_faces": stats.images_without_faces,
        "faces_detected": stats.faces_detected,
        "faces_embedded": stats.faces_embedded,
        "seconds": stats.seconds,
    }
    clear_names_and_montages()          # new photos -> new questions
    st.session_state["scan_stats"] = payload
    st.session_state["records"] = stats.records
    return payload


# -------------------------------------------------------------- clusters
def cluster_records(config: Dict, records: Sequence[FaceRecord]):
    """Cluster the cached records — cheap, so tolerance changes never
    trigger a re-scan."""
    return FaceClusterer.from_config(config).cluster(records)


def auto_names_for(clusters: Sequence[FaceCluster], config: Dict) -> Dict[int, str]:
    """Names the persistent DB recognises (M3), empty when unavailable."""
    if not clusters or not config.get("names_db"):
        return {}
    db = open_names_db(config)
    if db is None:
        return {}
    try:
        labels, _pending = cli.auto_label_clusters(
            clusters, db, float(config.get("tolerance", 0.5)))
        return labels
    finally:
        db.close()


def montage_for(cluster: FaceCluster):
    """Cached montage (PIL image) for one cluster."""
    key = f"montage_{cluster.cluster_id}_{cluster.size}"
    image = st.session_state.get(key)
    if image is None:
        image = build_montage(cluster, cell_size=CELL_SIZE)
        st.session_state[key] = image
    return image


def name_widget_key(cluster: FaceCluster) -> str:
    return f"{NAME_KEY_PREFIX}{cluster.cluster_id}"


def prefill_names(clusters: Sequence[FaceCluster],
                  auto_labels: Dict[int, str]) -> None:
    """Give every name input its starting value: the DB's guess, else ""."""
    for cluster in clusters:
        key = name_widget_key(cluster)
        if key not in st.session_state:
            st.session_state[key] = auto_labels.get(cluster.cluster_id, "")


def collect_names(clusters: Sequence[FaceCluster]) -> Dict[int, Optional[str]]:
    """Read the name inputs and normalise them for the organizer."""
    names: Dict[int, Optional[str]] = {}
    for cluster in clusters:
        raw = st.session_state.get(name_widget_key(cluster), "")
        names[cluster.cluster_id] = normalize_person_name(raw)
    return names


def persist_names(clusters: Sequence[FaceCluster],
                  names: Dict[int, Optional[str]], config: Dict) -> int:
    """Save newly typed names to the DB — only when they actually changed,
    and never for auto-detected ones (M3 rule)."""
    if not config.get("names_db"):
        return 0
    if "persisted_names" not in st.session_state:
        st.session_state["persisted_names"] = {}
    seen: Dict[int, str] = st.session_state["persisted_names"]
    db = open_names_db(config)
    if db is None:
        return 0
    saved = 0
    try:
        for cluster in clusters:
            name = names.get(cluster.cluster_id)
            if not name or seen.get(cluster.cluster_id) == name:
                continue
            if cli.save_named_cluster(db, cluster, name):
                seen[cluster.cluster_id] = name
                saved += 1
    finally:
        db.close()
    return saved


# ------------------------------------------------------------- rendering
def render_cluster_card(cluster: FaceCluster, auto_labels: Dict[int, str]) -> None:
    """Montage on the left, name input (pre-filled) on the right."""
    with st.container(border=True):
        left, right = st.columns([2, 3])
        with left:
            st.image(montage_for(cluster), width="stretch",
                     caption=f"{cluster.size} face(s) / "
                             f"{len(cluster.image_paths)} photo(s)")
        with right:
            auto = auto_labels.get(cluster.cluster_id)
            if auto:
                st.success(f"Cluster #{cluster.cluster_id} — "
                           f"auto-detected: **{auto}** (name database)")
            else:
                st.markdown(f"#### Cluster #{cluster.cluster_id}")
            prefill_names([cluster], auto_labels)
            st.text_input("Who is this? (name/skip)",
                          key=name_widget_key(cluster),
                          placeholder="Alice")
            with st.expander(f"{len(cluster.image_paths)} photo(s)"):
                for path in cluster.image_paths:
                    st.caption(str(path))


def render_summary(summary) -> None:
    if summary.files_placed:
        st.success(f"Placed **{summary.files_placed}** photo(s) into "
                   f"**{len(summary.folders)}** folder(s).")
    else:
        st.warning("Nothing was placed.")
    st.code(summary.format(), language="text")
    for source, reason in summary.errors:
        st.error(f"{source}: {reason}")


def _scan_section(config: Dict) -> Optional[Dict]:
    """Return the cached scan payload, running a new scan on demand."""
    if cached_scan_matches(config):
        return st.session_state["scan_stats"]

    stale = st.session_state.get("scan_stats") is not None
    if stale:
        st.warning("The input folder changed — press **Scan photos & "
                   "detect faces** to process the new folder.")
    if st.button("Scan photos & detect faces", key="scan",
                 type="primary", width="stretch"):
        try:
            return run_scan(config)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ModelsNotInstalledError as exc:
            st.error(f"{exc}\n\nRun `python -m src.main --fetch-models` "
                     "once to download the model weights.")
        except Exception as exc:  # never leave a broken stack trace only
            logger.exception("scan failed")
            st.error(f"Scan failed: {exc}")
    return None


def main() -> None:
    st.set_page_config(page_title="Face Grouping & Auto-Organizer",
                       page_icon="🧑", layout="wide")

    try:
        defaults = cli.load_config(CONFIG_PATH)
    except Exception as exc:  # missing config is not fatal for the GUI
        defaults = {}
        st.sidebar.warning(f"config.yaml not readable ({exc}) — "
                           "using built-in defaults")

    init_session(defaults)
    render_sidebar()
    config = current_config()

    st.title("Face Grouping & Auto-Organizer")
    st.caption("Offline face clustering — scan, review the groups, name "
               "them, then run the organizer.")

    scan = _scan_section(config)
    if scan is None:
        if not st.session_state.get("records"):
            st.info("Set the **input folder** in the sidebar, then press "
                    "**Scan photos & detect faces**.")
        return

    records = st.session_state["records"]
    if not records:
        st.warning("No readable photos were found in the input folder.")
        return

    clusters = cluster_records(config, records)

    # a changed grouping invalidates the names/montages of the old one
    signature = tuple(sorted(
        (cluster.cluster_id, tuple(str(p) for p in cluster.image_paths))
        for cluster in clusters
    ))
    if st.session_state.get("cluster_signature") != signature:
        st.session_state["cluster_signature"] = signature
        clear_names_and_montages()
    if not clusters:
        st.warning("No group is big enough to show (see "
                   "`min_faces_per_cluster` in config.yaml).")
        return

    auto_labels = auto_names_for(clusters, config)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Photos scanned", scan["images_scanned"])
    col2.metric("Faces detected", scan["faces_detected"])
    col3.metric("Clusters", len(clusters))
    col4.metric("Auto-labeled", len(auto_labels))
    if scan["seconds"]:
        st.caption(f"Scan took {scan['seconds']:.1f}s — cached, so typing "
                   "names and moving sliders never re-runs it.")

    st.subheader("Review the clusters")
    for cluster in clusters:
        render_cluster_card(cluster, auto_labels)

    st.divider()
    names = collect_names(clusters)
    unnamed = [cluster.cluster_id for cluster in clusters
               if names.get(cluster.cluster_id) is None]
    left, right = st.columns([3, 1])
    with left:
        st.caption(
            f"Mode **{config['mode']}** → `{config['output_folder']}` · "
            f"{len(clusters) - len(unnamed)} named, "
            f"{len(unnamed)} to the unknown folder"
        )
    with right:
        clicked = st.button("Run Organizer", key="run_organizer",
                            type="primary", width="stretch")

    if not clicked:
        previous = st.session_state.get("last_summary")
        if previous is not None:
            render_summary(previous)
        return

    if config["mode"] == "move":
        st.warning("**Move mode:** originals are removed from the input "
                   "folder after a successful copy.")
    try:
        organizer = PhotoOrganizer(
            output_folder=config["output_folder"],
            mode=config["mode"],
            unknown_folder=config["unknown_folder"],
        )
        summary = organizer.organize(
            [(names.get(cluster.cluster_id), cluster) for cluster in clusters]
        )
    except (ValueError, OSError) as exc:
        st.error(f"Could not organize the photos: {exc}")
        return

    remembered = persist_names(clusters, names, config)
    st.session_state["last_summary"] = summary
    if remembered:
        st.info(f"Remembered {remembered} name(s) in the local name "
                "database — next run will auto-detect them.")
    render_summary(summary)


if __name__ == "__main__":
    main()
