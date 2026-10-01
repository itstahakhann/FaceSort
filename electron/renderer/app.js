'use strict';

/**
 * FaceSort — renderer logic.
 *
 * No framework and no build step: the UI is driven straight off the five
 * engine endpoints through the preload bridge (`window.faceorg`).
 *
 * Flow:  ready → prefill from /status → Scan → poll /status for progress →
 *        GET /clusters → name groups (POST /name_cluster) → POST /organize.
 */

const bridge = window.faceorg;

const el = (id) => document.getElementById(id);
const ui = {
  input: el('input-folder'),
  output: el('output-folder'),
  tolerance: el('tolerance'),
  toleranceValue: el('tolerance-value'),
  minFaces: el('min-faces'),
  workers: el('workers'),
  scan: el('scan'),
  organize: el('organize'),
  pickInput: el('pick-input'),
  pickOutput: el('pick-output'),
  progressPanel: el('progress-panel'),
  progressBar: el('progress-bar'),
  progressLabel: el('progress-label'),
  progressCount: el('progress-count'),
  summaryPanel: el('summary-panel'),
  summary: el('summary'),
  logPanel: el('log-panel'),
  log: el('log'),
  logToggle: el('log-toggle'),
  title: el('content-title'),
  sub: el('content-sub'),
  chips: el('chips'),
  clusters: el('clusters'),
  empty: el('empty-state'),
  toast: el('toast'),
};

const state = {
  mode: 'copy',
  clusters: [],
  pollTimer: null,
  toastTimer: null,
  engineReady: false,
};

/* ------------------------------------------------------------- helpers */

function toast(message, kind = '') {
  ui.toast.textContent = message;
  ui.toast.className = `toast ${kind}`;
  ui.toast.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { ui.toast.hidden = true; }, 5200);
}

function setBusy(busy) {
  ui.scan.disabled = busy;
  ui.pickInput.disabled = busy;
  ui.pickOutput.disabled = busy;
  ui.scan.textContent = busy ? 'Scanning…' : 'Scan photos';
}

function readSettings() {
  return {
    input_folder: ui.input.value.trim(),
    output_folder: ui.output.value.trim(),
    tolerance: Number(ui.tolerance.value),
    min_faces_per_cluster: Number(ui.minFaces.value || 2),
    mode: state.mode,
    workers: Number(ui.workers.value || 0),
    use_db: true,
  };
}

function applyStatus(status) {
  if (!status) return;
  const config = status.config || {};
  // Only prefill a folder that really exists: on a fresh install the shipped
  // defaults (./input_photos) are placeholders, and showing a dead path in the
  // field is worse than showing our own placeholder text.
  if (!ui.input.value) {
    ui.input.value = config.input_exists ? config.input_folder : '';
  }
  if (!ui.output.value) {
    ui.output.value = config.output_exists ? config.output_folder : '';
  }
  if (!ui.tolerance.dataset.touched) {
    ui.tolerance.value = String(config.tolerance ?? 0.5);
    ui.toleranceValue.textContent = Number(ui.tolerance.value).toFixed(2);
  }
  if (!ui.minFaces.dataset.touched) {
    ui.minFaces.value = String(config.min_faces_per_cluster ?? 2);
  }
  if (!ui.workers.dataset.touched) {
    ui.workers.value = String(config.workers ?? 0);
  }
  setMode(config.mode === 'move' ? 'move' : 'copy', true);
  renderStats(status);
}

function renderStats(status) {
  const stats = status.stats || {};
  const rows = [];
  if (stats.photos !== undefined) {
    rows.push(['Photos scanned', stats.photos, '']);
    if (stats.without_faces) rows.push(['No faces found', stats.without_faces, 'warn']);
    if (stats.unreadable) rows.push(['Unreadable', stats.unreadable, 'warn']);
    if (stats.failed) rows.push(['Failed', stats.failed, 'danger']);
    rows.push(['Faces found', stats.faces ?? 0, '']);
    rows.push(['Groups', status.clusters ?? 0, '']);
    if (stats.seconds) {
      rows.push(['Time', `${stats.seconds.toFixed(1)} s (${stats.img_per_s ?? 0} img/s)`, '']);
    }
  }
  const results = status.results;
  if (results) {
    rows.push(['Sorted into', `${Object.keys(results.folders).length} folder(s)`, 'ok']);
    rows.push(['Photos placed', results.files_placed, 'ok']);
    if (results.errors && results.errors.length) {
      rows.push(['Errors', results.errors.length, 'danger']);
    }
  }
  ui.summary.replaceChildren();
  for (const [label, value, kind] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.className = kind;
    dd.textContent = String(value);
    ui.summary.append(dt, dd);
  }
  ui.summaryPanel.hidden = rows.length === 0;
}

function renderChips(status) {
  ui.chips.replaceChildren();
  const stats = status.stats || {};
  const chips = [
    ['Photos', stats.photos],
    ['Faces', stats.faces],
    ['Groups', status.clusters],
    ['Named', status.named],
  ].filter(([, value]) => value !== undefined && value !== null);
  for (const [label, value] of chips) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const strong = document.createElement('b');
    strong.textContent = String(value);
    chip.append(strong, document.createTextNode(` ${label.toLowerCase()}`));
    ui.chips.append(chip);
  }
}

/* ------------------------------------------------------------- scanning */

async function startScan() {
  const settings = readSettings();
  if (!settings.input_folder || !settings.output_folder) {
    toast('Choose an input and an output folder first.', 'error');
    return;
  }
  state.clusters = [];
  ui.clusters.replaceChildren();
  ui.empty.hidden = false;
  setBusy(true);
  try {
    await bridge.api.scan(settings);
    ui.progressPanel.hidden = false;
    ui.progressBar.style.width = '0%';
    ui.progressLabel.textContent = 'Starting…';
    ui.progressCount.textContent = '0 / 0';
    pollStatus();
  } catch (error) {
    setBusy(false);
    toast(String(error.message || error), 'error');
  }
}

function pollStatus() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    let status;
    try {
      status = await bridge.api.status();
    } catch (error) {
      ui.progressLabel.textContent = 'Lost contact with the engine…';
      pollStatus();
      return;
    }
    applyStatus(status);
    renderChips(status);

    if (status.scanning) {
      const total = status.total || 0;
      const done = status.processed || 0;
      const percent = total ? Math.min(100, (done / total) * 100) : 0;
      ui.progressBar.style.width = `${percent}%`;
      ui.progressCount.textContent = `${done} / ${total}`;
      ui.progressLabel.textContent = status.current
        ? `Scanning ${status.current}`
        : (status.phase || 'Scanning…');
      pollStatus();
      return;
    }

    setBusy(false);
    if (status.state === 'error') {
      ui.progressLabel.textContent = 'Scan failed';
      toast(status.error || 'The scan failed.', 'error');
      return;
    }
    if (status.state === 'ready' || status.state === 'done') {
      ui.progressBar.style.width = '100%';
      ui.progressLabel.textContent = status.phase || 'Ready';
      await loadClusters();
    }
  }, 450);
}

async function loadClusters() {
  try {
    const payload = await bridge.api.clusters();
    state.clusters = payload.clusters || [];
    renderClusters(state.clusters);
    ui.empty.hidden = state.clusters.length > 0;
    ui.title.textContent = 'Review the groups';
    ui.sub.textContent = state.clusters.length
      ? `${state.clusters.length} group(s) — name each one, then sort.`
      : 'No faces were found in this folder.';
    ui.organize.disabled = state.clusters.length === 0;
  } catch (error) {
    toast(String(error.message || error), 'error');
  }
}

/* -------------------------------------------------------------- clusters */

function renderClusters(clusters) {
  ui.clusters.replaceChildren();
  for (const cluster of clusters) {
    ui.clusters.append(buildCard(cluster));
  }
}

function buildCard(cluster) {
  const card = document.createElement('article');
  card.className = 'card';

  const head = document.createElement('div');
  head.className = 'card-head';
  const title = document.createElement('span');
  title.className = 'card-title';
  title.textContent = `Group ${cluster.id + 1}`;
  const badge = document.createElement('span');
  badge.className = `badge ${cluster.name ? (cluster.auto ? 'auto' : '') : 'unnamed'}`;
  badge.textContent = cluster.name
    ? (cluster.auto ? 'remembered' : 'named')
    : 'not named';
  head.append(title, badge);

  const meta = document.createElement('span');
  meta.className = 'card-meta';
  meta.textContent = `${cluster.size} face${cluster.size === 1 ? '' : 's'} in `
    + `${cluster.photos} photo${cluster.photos === 1 ? '' : 's'}`;

  const faces = document.createElement('div');
  faces.className = 'faces';
  for (const face of cluster.faces) {
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'face';
    tile.title = `${face.photo} — click to enlarge`;
    if (face.thumb) {
      const image = document.createElement('img');
      image.src = face.thumb;
      image.alt = `Face from ${face.photo}`;
      tile.append(image);
    }
    const caption = document.createElement('span');
    caption.className = 'face-name';
    caption.textContent = face.photo;
    tile.append(caption);
    tile.addEventListener('click', () => openLightbox(face));
    faces.append(tile);
  }

  const row = document.createElement('div');
  row.className = 'card-name-row';
  const input = document.createElement('input');
  input.type = 'text';
  input.value = cluster.name || '';
  input.placeholder = 'Name this person…';
  input.setAttribute('aria-label', `Name for group ${cluster.id + 1}`);
  const save = document.createElement('button');
  save.type = 'button';
  save.className = 'primary';
  save.textContent = 'Save';
  save.addEventListener('click', () => saveName(cluster, input.value, badge, card));
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      saveName(cluster, input.value, badge, card);
    }
  });
  row.append(input, save);

  card.append(head, meta, faces, row);
  return card;
}

async function saveName(cluster, value, badge, card) {
  try {
    const result = await bridge.api.nameCluster(cluster.id, value.trim());
    const named = Boolean(result.name);
    badge.className = `badge ${named ? (result.remembered ? 'auto' : '') : 'unnamed'}`;
    badge.textContent = named
      ? (result.remembered ? 'remembered' : 'named')
      : 'not named';
    toast(named ? `Group ${cluster.id + 1} → ${result.name}` :
                   `Group ${cluster.id + 1} will go to the unknown folder`, 'ok');
  } catch (error) {
    toast(String(error.message || error), 'error');
  }
}

function openLightbox(face) {
  const box = document.createElement('div');
  box.className = 'lightbox';
  const image = document.createElement('img');
  if (face.thumb) image.src = face.thumb;
  const caption = document.createElement('div');
  caption.className = 'caption';
  caption.textContent = `${face.photo}  ·  ${face.path}`;
  box.append(image, caption);
  box.addEventListener('click', () => box.remove());
  document.addEventListener('keydown', function onKey(event) {
    if (event.key === 'Escape') {
      box.remove();
      document.removeEventListener('keydown', onKey);
    }
  });
  document.body.append(box);
}

/* ------------------------------------------------------------- organize */

async function runOrganize() {
  setBusy(true);
  ui.organize.disabled = true;
  // The engine snapshots the names when /organize starts, so lock the fields
  // while it copies — otherwise an edit would look like it took effect but
  // not appear in the folders it is sorting right now.
  const nameInputs = [...document.querySelectorAll('.card input, .card .primary')];
  nameInputs.forEach((input) => { input.disabled = true; });
  try {
    const results = await bridge.api.organize(state.mode);
    const folders = Object.entries(results.folders)
      .map(([name, count]) => `${name} (${count})`).join(', ');
    toast(`Sorted ${results.files_placed} photo(s) into ${folders || 'no folders'}`,
          results.ok ? 'ok' : 'error');
    const status = await bridge.api.status();
    applyStatus(status);
    renderChips(status);
  } catch (error) {
    toast(String(error.message || error), 'error');
  } finally {
    setBusy(false);
    nameInputs.forEach((input) => { input.disabled = false; });
    ui.organize.disabled = state.clusters.length === 0;
  }
}

/* ------------------------------------------------------------------ logs */

function appendLog(line) {
  ui.logPanel.hidden = false;
  ui.log.textContent += `${line}\n`;
  if (ui.log.textContent.length > 20000) {
    ui.log.textContent = ui.log.textContent.slice(-16000);
  }
  ui.log.scrollTop = ui.log.scrollHeight;
}

/* -------------------------------------------------------------- wiring */

function setMode(mode, quiet = false) {
  state.mode = mode;
  for (const button of document.querySelectorAll('.seg')) {
    const active = button.dataset.mode === mode;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-checked', active ? 'true' : 'false');
  }
  if (!quiet) toast(`Sorting mode: ${mode}`);
}

function wire() {
  ui.tolerance.addEventListener('input', () => {
    ui.tolerance.dataset.touched = '1';
    ui.toleranceValue.textContent = Number(ui.tolerance.value).toFixed(2);
  });
  for (const field of [ui.minFaces, ui.workers]) {
    field.addEventListener('input', () => { field.dataset.touched = '1'; });
  }
  for (const button of document.querySelectorAll('.seg')) {
    button.addEventListener('click', () => setMode(button.dataset.mode));
  }

  ui.pickInput.addEventListener('click', async () => {
    const folder = await bridge.pickFolder('input');
    if (folder) ui.input.value = folder;
  });
  ui.pickOutput.addEventListener('click', async () => {
    const folder = await bridge.pickFolder('output');
    if (folder) ui.output.value = folder;
  });

  ui.scan.addEventListener('click', startScan);
  ui.organize.addEventListener('click', runOrganize);
  ui.logToggle.addEventListener('click', () => {
    ui.log.hidden = !ui.log.hidden;
    ui.logToggle.textContent = ui.log.hidden ? 'show' : 'hide';
  });

  bridge.onLog(appendLog);
  bridge.onPickInputRequested(async () => {
    const folder = await bridge.pickFolder('input');
    if (folder) ui.input.value = folder;
  });

  bridge.onReady(async (payload) => {
    state.engineReady = true;
    applyStatus(payload.status);
    renderChips(payload.status || {});
    try {
      const status = await bridge.api.status();
      applyStatus(status);
      // A scan from a previous session is still valid data to show.
      if (status.state === 'ready' || status.state === 'done') await loadClusters();
    } catch (error) {
      /* engine not answering yet; the scan button reports the real error */
    }
    if (bridge.smoke) bridge.smoke();
  });
}

wire();
