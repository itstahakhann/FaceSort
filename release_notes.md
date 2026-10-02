FaceSort groups photos by whose face is in them and files each one into a
folder named after that person. Everything runs on your own machine — no photo,
face or name is ever uploaded.

The problem it solves is age progression. Standard face embeddings encode face
*shape* as much as identity, and shape changes as a child grows, so one person's
childhood and adult photos land in separate groups. FaceSort adds a second
fingerprint from the periocular region — brows, eyes and the bridge of the nose,
which barely change with age — and clusters on a blend of both.

## Downloads (Windows x64)

| File | Size | What it is |
|------|------|------------|
| `FaceSort-Setup-1.0.0.exe` | 386 MB | Installer. Puts FaceSort in your Start menu. |
| `FaceSort-Portable-1.0.0.exe` | 360 MB | Single file. Runs from anywhere, no install. |

> **These builds are not code-signed.** Windows SmartScreen will show a blue
> "Windows protected your PC" warning on first run, because it cannot verify an
> unknown publisher. Click **More info → Run anyway**. This is expected for any
> self-built open-source release, not a sign of a problem with the download.

## First run

1. Launch it and drop in (or browse to) the folder of photos you want sorted.
2. Pick an output folder, then **Scan photos**.
3. Review the groups and name each person.
4. **Sort into folders** copies the photos into `output/<person>/`.

You can also drag a folder anywhere onto the window.

### If one person appears as two groups

Usually a childhood photo and an adult one that the model could not bridge —
often a blurry or very small face. Click **Same person?**, select the two
groups, and link them. Give it a name and the link is remembered, so future
scans group them on their own.

## Requirements and performance

- Windows 10/11, x64. No GPU needed, no Python, no internet.
- Roughly **0.3 images per second** on a modest dual-core laptop, rising with
  more cores. A 1,000-photo library takes around an hour.
- The age-invariance pass adds about 21% per image; detection dominates the
  cost either way.

## Privacy

Inference is fully offline. The only code path that touches the network is
`python build.py models`, which fetches the ONNX weights once during a build —
never during normal use. Your photos are read, never modified (unless you
choose **Move** instead of **Copy**).

## Building from source

```
git clone https://github.com/itstahakhann/FaceSort
cd FaceSort
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python build.py models
build.bat
```

MIT licensed — see [LICENSE](https://github.com/itstahakhann/FaceSort/blob/main/LICENSE).