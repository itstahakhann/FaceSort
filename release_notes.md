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

Three sections sit along the top: **Organise**, **People together** and
**Share gallery**.

1. In **Organise**, press **Choose photo folder** (or drag a folder anywhere
   onto the window). Where the sorted folders go is shown inline on that first
   screen; everything else lives behind *Change settings*.
2. Watch the count climb while it reads your library.
3. On **Name each group**, give each person a name. Click a face to enlarge
   it, or *Details* to see exactly which photos and folder they are headed for.
4. **Sort into folders** copies the photos into `output/<person>/`. You can
   watch them go: the folders appear as they are created and each photo flies
   into the right one.

Groups you skip go into `_unknown`; photos with no face go into `_no_faces`.
Nothing is ever discarded.

### If one person appears as two groups

Usually a childhood photo and an adult one that the model could not bridge —
often a blurry or very small face. Click **Same person?**, select the two
groups, and link them. Give it a name and the link is remembered, so future
scans group them on their own.

## Find photos of people together

**People together** answers "who was in this photo?". Pick two or more people
and you get only the photos where *every* one of them appears — which is how
you find the family trip rather than one person's pictures of it. A single
person shows all their photos. There is also a co-occurrence map where darker
means more shared photos, and clicking a pair searches for those two.

It draws on runs where you typed a name, so name your groups as you go.

## Share a gallery

**Share gallery** writes a folder containing one HTML page, a stylesheet, a
script and your photos. It opens in any browser with no server and no network,
so it can be zipped, emailed, put on a USB stick, or opened on a phone with no
signal — and it will still work in ten years. It has a people index, a
per-person grid, a full-screen lightbox (arrow keys, `Esc`, swipe), a name
filter, and a dark/light toggle.

You can restrict it to chosen people and put a password on it.

**The password is a gate, not encryption.** The SHA-256 digest ships inside the
HTML, so anyone who opens the source can read it, and a weak password falls to
an offline dictionary. It stops a family member or a guest from casually
browsing; it does not stop anyone who means it — and because a browser must be
able to read a JPEG to display it, anyone who can open the page can open the
image files directly. If you need real protection, keep the photos somewhere
the browser cannot reach.

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