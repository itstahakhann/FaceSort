# Face Grouping & Auto-Organizer — Software Specification

## 1. Purpose
A local, offline desktop tool that scans a folder of photos, detects all
human faces, groups photos of the same person together, asks the user to
name each group, and finally sorts the photos into named folders.

## 2. Goals
- Zero cloud dependency (privacy — photos never leave the machine)
- Zero monetary cost (only open-source libraries)
- Works on CPU-only machines
- Simple enough for a non-technical user to run

## 3. Non-Goals
- Real-time webcam recognition
- Face generation, editing, or deepfake detection
- Cloud sync or mobile app (initial version)

## 4. Core User Flow
1. User launches the app.
2. User selects an input folder containing photos.
3. App scans every image, detects faces, and computes a face embedding
   (numeric fingerprint) for each detected face.
4. App clusters embeddings so that all faces of the same person fall into
   the same cluster.
5. For each cluster, app shows a thumbnail grid of sample faces and asks:
   "Who is this?" User types a name (or "skip" / "unknown").
6. App copies (or moves) every photo containing that person into a
   subfolder named after them.
7. Photos with multiple named people appear in multiple folders.
8. Unnamed clusters go into an `_unknown/` folder.
9. A summary report is shown: total photos scanned, faces detected,
   clusters formed, folders created.

## 5. Functional Requirements

| ID  | Requirement |
|-----|-------------|
| F1  | Support image formats: jpg, jpeg, png, bmp, webp, heic (optional) |
| F2  | Detect 0..N faces per image (multiple faces per photo allowed) |
| F3  | Generate a face embedding per detected face |
| F4  | Cluster embeddings using a distance threshold (configurable) |
| F5  | Show a visual preview per cluster before naming |
| F6  | Accept a user-provided name per cluster |
| F7  | Create one folder per name under an output directory |
| F8  | Copy or move original photos into the matching folder(s) |
| F9  | Skip clusters the user marks as irrelevant |
| F10 | Produce a log/summary of the operation |
| F11 | Persist known names so future runs auto-label familiar people |
| F12 | Handle duplicate filenames safely (append suffix) |

## 6. Non-Functional Requirements
- **Privacy:** All processing local. No network calls.
- **Performance:** Target ≥ 5 images/sec on CPU for 1080p photos.
- **Robustness:** Skip corrupt/unreadable images without crashing.
- **Portability:** Windows, macOS, Linux.
- **Usability:** Single command or double-click to launch.
- **Configurability:** Tolerance, min photos per cluster, input/output
  paths, copy-vs-move mode, all set via a config file or CLI flags.

## 7. Architecture (High Level)

    [ Input Folder ]
           |
           v
    [ Image Loader ]  --(skip bad files)--> [ Log ]
           |
           v
    [ Face Detector ]  (RetinaFace / dlib HOG or CNN)
           |
           v
    [ Embedding Extractor ]  (InsightFace buffalo_l / dlib 128-D)
           |
           v
    [ Clustering Engine ]  (DBSCAN, cosine or euclidean metric)
           |
           v
    [ Preview Generator ]  (thumbnail montage per cluster)
           |
           v
    [ User Naming UI ]  (CLI prompt / tkinter / streamlit)
           |
           v
    [ File Organizer ]  --> [ Named Output Folders ]
           |
           v
    [ Persistent Name DB ]  (SQLite / JSON)

## 8. Data Model
- **FaceRecord**: image_path, bbox(top,right,bottom,left), embedding[128],
  cluster_id, person_name
- **Person**: name, embedding_centroid, sample_image_paths[]
- **RunLog**: timestamp, input_folder, output_folder, counts

## 9. Configuration Options
| Key | Default | Description |
|-----|---------|-------------|
| input_folder | ./input_photos | Source of images |
| output_folder | ./grouped_photos | Destination |
| tolerance | 0.5 | Clustering strictness |
| min_faces_per_cluster | 2 | Ignore tiny clusters |
| mode | copy | copy or move |
| detector_backend | retinaface | retinaface / dlib_hog / dlib_cnn |
| unknown_folder | _unknown | Folder for skipped clusters |

## 10. Edge Cases to Handle
- Image with no faces → leave in `_no_faces/` or ignore (configurable)
- Same face detected twice in one image → deduplicate by IoU
- Two different people merged into one cluster → allow split/reassign
- Same person split into two clusters → allow merge
- Name collision (folder already exists) → reuse existing folder
- Very large folders (>10k images) → batch processing + progress bar
- HEIC/RAW files → optional plugin

## 11. Milestones
1. **M1 – Core CLI:** scan → detect → cluster → prompt → sort. (no GUI)
2. **M2 – Preview UI:** thumbnail montage per cluster in terminal or popup.
3. **M3 – Persistent DB:** remember names across runs.
4. **M4 – GUI:** tkinter or Streamlit front-end.
5. **M5 – Packaging:** PyInstaller standalone executable.
6. **M6 – Performance:** optional GPU path + multiprocessing.

## 12. Acceptance Criteria
- Given 100 mixed photos with 3 people, the tool creates 3 named folders
  containing the correct photos with ≥ 90% accuracy on default settings.
- No photo is deleted or modified without user confirmation.
- User can rename, merge, or split clusters during review.
- Runs fully offline; no outbound network traffic.

## 13. Suggested Repo Layout

    face-organizer/
      ├── src/
      │   ├── detector.py
      │   ├── embedder.py
      │   ├── clusterer.py
      │   ├── organizer.py
      │   ├── names_db.py
      │   └── ui/
      ├── tests/
      ├── config.yaml
      ├── requirements.txt
      ├── SPEC.md
      └── README.md

## 14. Dependencies (All Free)
- insightface, onnxruntime (CPU)
- face_recognition, dlib (alternative)
- opencv-python, Pillow, numpy, scikit-learn
- streamlit or tkinter (UI)
- pyinstaller (packaging)

## 15. Open Questions for the Implementer
- Should the tool support video files too?
- Should it auto-suggest names using a pre-trained DB of the user's friends?
- Should we add EXIF-based date sorting as a secondary dimension?
- Do we want a web version (Flask) or keep it desktop-only?