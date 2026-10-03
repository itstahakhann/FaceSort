# FaceFlow 1.0.0

**Private photo organization, powered locally.**

FaceFlow finds the people in your photo library and files each photo into a
folder named after them. Everything runs on your own machine — no photo, face or
name is ever uploaded, and the app makes no network requests at all. There is
no account and nothing to sign in to.

## What it does that a plain face-recognition script does not

The hard problem is **age progression**. Standard face embeddings encode face
*shape* as much as identity, and shape changes as a child grows, so one person's
childhood photos and adult photos land in separate groups. FaceFlow adds a
second fingerprint from the periocular region — brows, eyes and the bridge of
the nose, which barely change with age — and clusters on a blend of both.

When it still cannot bridge the gap — usually one blurry or very small photo —
you correct it yourself. Merge two groups that are really the same person, give
it a name, and the link is remembered so future scans group them on their own.

## Downloads (Windows x64)

| File | Size | What it is |
|------|------|------------|
| `FaceFlow-Setup-1.0.0.exe` | SIZE_SETUP | Installer. Adds FaceFlow to your Start menu. |
| `FaceFlow-Portable-1.0.0.exe` | SIZE_PORTABLE | Single file. Runs from anywhere, no install. |

> **These builds are not code-signed.** Windows SmartScreen will show a blue
> "Windows protected your PC" warning on first run, because it cannot verify an
> unknown publisher. Click **More info → Run anyway**. This is expected for any
> self-built open-source release, not a sign of a problem with the download.

## Getting started

Five sections down the left, and the content area changes without the window
ever reloading.

1. **Overview** tells you what has been found so far and what to do next. Press
   **Scan photos**, or drag a photo folder anywhere onto the window.
2. **Photos** is the screen that matters. Every person-sized group is a card
   with their face montage, their photo count, and four actions: *Name*,
   *Merge*, *Split* and *Details*. Give each group a name, then **Sort into
   folders** when you are done.

Groups you skip go into `_unknown` and photos with no face go into
`_no_faces`. Nothing is ever discarded.

### If one person appears as two groups

Click **Merge** on one of them, then click the other. Their photos are combined
into a single group. Supply a name and the link is remembered permanently.

### Split is not available in 1.0.0

The **Split** control is present but disabled, and it says why: the local engine
has no split operation yet. Use **Merge** to join two groups the engine pulled
apart, and **Skip** to set a group aside. Adding real splitting means changing
the recognition and storage layers, not just the interface, so it is deliberately
not faked here.

## People

Everyone you have named, with their photo count. Open one to see their photos
in a grid, rename them, or jump straight to finding who they were photographed
with.

## Relationships

**Pick two or more people to see only the photos where every one of them
appears** — which is how you find the family trip rather than one person's
pictures of it. A single person shows all their photos.

Below that, *Who appears together* ranks pairs by how many photos they share.
Click a pair to see them. Export any result to a new folder.

This draws on the names you give your groups, so it grows as you use the app.

## Gallery export

Builds a folder containing one HTML page, a stylesheet, a script and your photos.
It opens in any browser with no server and no network, so you can zip it, email
it, put it on a USB stick, or open it on a phone with no signal — and it will
still work in ten years.

It has a people index, a per-person grid, a full-screen viewer (arrow keys,
`Esc`, swipe on a phone), a name filter and a dark/light toggle. You can restrict
it to chosen people and put a password on it.

**The password is a gate, not encryption.** The SHA-256 digest ships inside the
HTML file, so anyone who opens the source can read it, and a weak password falls
to an offline dictionary. It stops a family member or a guest from casually
browsing; it does not stop anyone determined — and because a browser must be able
to read a JPEG to display it, anyone who can open the page can open the image
files directly. FaceFlow says this in the app as well, next to the setting.

## Settings

Everything the engine actually reads, grouped as **Scanning**,
**Organization** and **Appearance**: photo folder, destination, minimum faces
per group, worker processes, match strictness, copy-vs-move, and theme.

Settings are applied to your next scan. FaceFlow does not claim to save them
anywhere else, because it does not.

## Requirements and performance

- Windows 10/11, x64. No GPU needed, no Python, no internet, no account.
- Roughly **0.3 images per second** on a modest dual-core laptop, rising with
  more cores. A 1,000-photo library takes around an hour.
- The age-invariance pass adds about 21% per image; detection dominates the
  cost either way.
- **Worker processes** in Settings will finish sooner at the cost of memory.
  It is set to auto by default.

## Privacy

Inference is fully offline. The only code path that touches the network is
`python build.py models`, which fetches the ONNX weights once during a build —
never during normal use.

Your photos are read, never modified, unless you choose **Move** instead of
**Copy**. There is no telemetry, no analytics, no crash reporting and no update
check. The offline indicator in the sidebar reflects the real network state, so
if it says offline, the app is genuinely not reaching the internet.

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