'use strict';

/**
 * FaceSort — renderer logic.
 *
 * No framework, no build step: the UI is a small state machine over the five
 * engine endpoints (`/status`, `/scan`, `/clusters`, `/name_cluster`,
 * `/organize`, plus the `/thumb` preview) reached through the preload bridge.
 *
 * Views: configure -> scanning -> review -> done.  Polling drives the scan and
 * organize phases; everything else is event-driven.
 */

const bridge = window.faceorg;

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const ui = {
  app: $('app'),
  sidebar: $('sidebar'),
  drawerToggle: $('drawer-toggle'),
  enginePill: $('engine-pill'),
  engineLabel: $('engine-label'),
  themeToggle: $('theme-toggle'),
  progressFill: $('progress-fill'),

  input: $('input-folder'),
  output: $('output-folder'),
  tolerance: $('tolerance'),
  toleranceValue: $('tolerance-value'),
  minFaces: $('min-faces'),
  workers: $('workers'),
  pickInput: $('pick-input'),
  pickOutput: $('pick-output'),

  scan: $('scan'),
  scanLabel: $('scan-label'),
  organize: $('organize'),
  organizeLabel: $('organize-label'),

  mini: $('mini-progress'),
  miniBar: $('mini-bar'),
  miniLabel: $('mini-label'),
  miniCount: $('mini-count'),
  summaryPanel: $('summary-panel'),
  summary: $('summary'),
  logPanel: $('log-panel'),
  log: $('log'),
  logToggle: $('log-toggle'),

  stage: $('stage'),
  dropzone: $('dropzone'),
  dropzoneChosen: $('dropzone-chosen'),
  dragVeil: $('drag-veil'),
  scanBar: $('scan-bar'),
  scanBarWrap: $('scan-bar-wrap'),
  scanPercent: $('scan-percent'),
  scanText: $('scan-text'),
  scanStats: $('scan-stats'),
  previewImg: $('preview-img'),
  previewName: $('preview-name'),
  clusters: $('clusters'),
  clusterSearch: $('cluster-search'),
  autoCelebrate: $('auto-celebrate'),
  linkToggle: $('link-toggle'),
  mergeTray: $('merge-tray'),
  mergeSlotA: $('merge-slot-a'),
  mergeSlotB: $('merge-slot-b'),
  mergeName: $('merge-name'),
  mergeConfirm: $('merge-confirm'),
  mergeCancel: $('merge-cancel'),
  doneStats: $('done-stats'),
  folderList: $('folder-list'),
  doneOpen: $('done-open'),
  doneAgain: $('done-again'),
  doneCelebrate: $('done-celebrate'),
  scrollTop: $('scroll-top'),

  modalLayer: $('modal-layer'),
  modalThumbs: $('modal-thumbs'),
  modalTitle: $('modal-title'),
  modalSub: $('modal-sub'),
  modalName: $('modal-name'),
  modalError: $('modal-error'),
  modalSuggestions: $('modal-suggestions'),
  modalSave: $('modal-save'),
  modalSkip: $('modal-skip'),

  toasts: $('toasts'),
  live: $('live'),
  confetti: $('confetti'),
};

const state = {
  view: 'configure',
  mode: 'copy',
  poll: null,
  pollBusy: false,
  clusters: [],
  names: new Map(),
  current: null,          // cluster being named in the modal
  lastFocus: null,
  lastPreview: '',
  busy: false,
  outputFolder: '',
  celebrate: true,
  /** Manual age bridge ("same person?"). */
  linking: false,
  linkPicks: [],           // cluster ids chosen, in order; max 2
};

/* ====================================================== tiny helpers */

function icon(name, cls) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', `icon ${cls || ''}`.trim());
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

function announce(text) {
  ui.live.textContent = text;
}

/** `plural(1, 'photo')` -> "1 photo", `plural(3, 'photo')` -> "3 photos". */
function plural(count, word) {
  return `${count} ${word}${count === 1 ? '' : 's'}`;
}

function toast(message, kind = '', iconName = null) {
  const node = document.createElement('div');
  node.className = `toast ${kind}`.trim();
  node.append(icon(iconName || (kind === 'error' ? 'alert' : kind === 'ok' ? 'check' : 'tag')));
  const text = document.createElement('span');
  text.textContent = message;
  node.append(text);
  ui.toasts.append(node);
  announce(message);
  const life = kind === 'error' ? 7000 : 3800;
  setTimeout(() => {
    node.classList.add('is-leaving');
    setTimeout(() => node.remove(), 240);
  }, life);
}

/* click ripple on every button */
document.addEventListener('pointerdown', (event) => {
  const button = event.target.closest('.btn, .icon-btn, .seg, .suggestion');
  if (!button || button.disabled) return;
  const rect = button.getBoundingClientRect();
  const span = document.createElement('span');
  span.className = 'ripple';
  const size = Math.max(rect.width, rect.height) * 2;
  span.style.width = span.style.height = `${size}px`;
  span.style.left = `${event.clientX - rect.left}px`;
  span.style.top = `${event.clientY - rect.top}px`;
  button.append(span);
  setTimeout(() => span.remove(), 620);
});

/* ============================================================= theme */

const THEME_KEY = 'facesort.theme';

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch (error) { /* private mode */ }
}

function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem(THEME_KEY); } catch (error) { /* ignore */ }
  const preferred = window.matchMedia('(prefers-color-scheme: light)').matches
    ? 'light' : 'dark';
  applyTheme(stored || preferred);
  ui.themeToggle.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    toast(`${next === 'dark' ? 'Dark' : 'Light'} theme`, '', 'sparkles');
  });
}

/* ============================================================== views */

const STEP_ORDER = ['configure', 'scanning', 'review', 'done'];

function setView(view) {
  if (state.view === view) return;
  state.view = view;
  for (const section of document.querySelectorAll('.view')) {
    const active = section.dataset.view === view;
    section.classList.toggle('is-active', active);
    section.hidden = !active;
  }
  const index = STEP_ORDER.indexOf(view);
  for (const step of document.querySelectorAll('.pstep')) {
    const stepIndex = STEP_ORDER.indexOf(step.dataset.step);
    step.classList.toggle('is-current', stepIndex === index);
    step.classList.toggle('is-done', stepIndex < index);
  }
  // one rail that fills to the active step — progress as a quantity, not decoration
  const reached = STEP_ORDER.length > 1 ? index / (STEP_ORDER.length - 1) : 0;
  ui.progressFill.style.width = `${reached * 100}%`;
  ui.stage.scrollTop = 0;
  updateScrollTop();
}

function setEngine(stateName, label) {
  ui.enginePill.dataset.state = stateName;
  ui.engineLabel.textContent = label;
}

function setBusy(busy, label) {
  state.busy = busy;
  ui.scan.disabled = busy;
  ui.organize.disabled = busy || state.clusters.length === 0;
  ui.pickInput.disabled = busy;
  ui.pickOutput.disabled = busy;
  ui.scanLabel.textContent = busy ? 'Working…' : 'Scan photos';
  if (label) ui.organizeLabel.textContent = label;
}

/** Keep the target's trailing slot in sync with the chosen folder. */
function showChosenFolder(path) {
  if (!ui.dropzoneChosen) return;
  const name = path ? path.split(/[\\/]/).filter(Boolean).pop() : '';
  ui.dropzoneChosen.textContent = name || '';
  ui.dropzone.classList.toggle('has-folder', Boolean(name));
}

/* ========================================================== settings */

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
  if (!ui.input.value) {
    ui.input.value = config.input_exists ? config.input_folder : '';
    if (ui.input.value) showChosenFolder(ui.input.value);
  }
  if (!ui.output.value) {
    ui.output.value = config.output_exists ? config.output_folder : '';
  }
  if (!ui.tolerance.dataset.touched) {
    ui.tolerance.value = String(config.tolerance ?? 0.5);
    syncRange();
  }
  if (!ui.minFaces.dataset.touched) ui.minFaces.value = String(config.min_faces_per_cluster ?? 2);
  if (!ui.workers.dataset.touched) ui.workers.value = String(config.workers ?? 0);
  setMode(config.mode === 'move' ? 'move' : 'copy', true);
  renderSummary(status);
}

function syncRange() {
  const input = ui.tolerance;
  const pct = ((Number(input.value) - Number(input.min)) /
               (Number(input.max) - Number(input.min))) * 100;
  input.style.setProperty('--pct', `${pct}%`);
  ui.toleranceValue.textContent = Number(input.value).toFixed(2);
}

function setMode(mode, quiet) {
  state.mode = mode;
  for (const button of document.querySelectorAll('.seg')) {
    const active = button.dataset.mode === mode;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-checked', active ? 'true' : 'false');
  }
  if (!quiet) ui.organizeLabel.textContent =
    mode === 'move' ? 'Move into folders' : 'Sort into folders';
}

function renderSummary(status) {
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
      rows.push(['Time', `${stats.seconds.toFixed(1)} s · ${stats.img_per_s ?? 0} img/s`, '']);
    }
  }
  const results = status.results;
  if (results) {
    rows.push(['Folders', Object.keys(results.folders).length, 'ok']);
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

function renderScanStats(stats) {
  const rows = [
    ['photos found', stats.photos], ['faces', stats.faces], ['groups', stats.groups],
  ].filter(([, value]) => value !== undefined && value !== null);
  ui.scanStats.replaceChildren();
  for (const [label, value] of rows) {
    const item = document.createElement('li');
    const strong = document.createElement('b');
    strong.textContent = String(value);
    item.append(strong, document.createTextNode(label));
    ui.scanStats.append(item);
  }
}

/* =========================================================== scanning */

async function startScan() {
  const settings = readSettings();
  if (!settings.input_folder || !settings.output_folder) {
    toast('Choose an input and an output folder first.', 'warn', 'alert');
    (settings.input_folder ? ui.output : ui.input).focus();
    return;
  }
  state.outputFolder = settings.output_folder;
  state.clusters = [];
  state.names.clear();
  state.linkPicks = [];
  ui.clusters.replaceChildren();
  setLinkMode(false);
  setView('scanning');
  setBusy(true);
  ui.previewImg.removeAttribute('src');
  ui.previewName.textContent = '—';
  ui.scanPercent.textContent = '0%';
  ui.scanBar.style.width = '0%';
  renderScanStats({});

  try {
    await bridge.api.scan(settings);
    setEngine('busy', 'scanning');
    clearTimeout(state.poll);      // never run two poll chains at once
    poll();
  } catch (error) {
    setBusy(false);
    setEngine('error', 'idle');
    setView('configure');
    toast(String(error.message || error), 'error');
  }
}

async function poll() {
  if (state.pollBusy) return;
  state.pollBusy = true;
  let status = null;
  try {
    status = await bridge.api.status();
  } catch (error) {
    state.pollBusy = false;
    state.poll = setTimeout(poll, 700);
    return;
  }
  state.pollBusy = false;
  applyStatus(status);

  if (status.scanning) {
    setView('scanning');
    setEngine('busy', 'scanning');
    const total = status.total || 0;
    const done = status.processed || 0;
    const pct = total ? Math.min(100, (done / total) * 100) : 0;
    ui.scanBar.style.width = `${pct}%`;
    ui.scanBarWrap.setAttribute('aria-valuenow', String(Math.round(pct)));
    ui.scanPercent.textContent = `${Math.round(pct)}%`;
    ui.scanText.textContent = total
      ? `Processing image ${done} of ${total} — ${status.current || '…'}`
      : (status.phase || 'Scanning…');
    ui.mini.hidden = false;
    ui.miniBar.style.width = `${pct}%`;
    ui.miniCount.textContent = `${done} / ${total}`;
    ui.miniLabel.textContent = 'Scanning photos';
    showPreview(status.current);
    announce(`Scanning ${done} of ${total}`);
    state.poll = setTimeout(poll, 400);
    return;
  }

  if (status.organizing) {
    setView('review');
    setEngine('busy', 'sorting');
    const info = status.organize || {};
    const total = info.total || 0;
    const done = info.done || 0;
    const pct = total ? Math.min(100, (done / total) * 100) : 0;
    ui.mini.hidden = false;
    ui.miniBar.style.width = `${pct}%`;
    ui.miniCount.textContent = `${done} / ${total}`;
    ui.miniLabel.textContent = `Sorting ${info.current || 'photos'}`;
    // a toast per poll tick would bury the UI — the meter already shows it
    state.poll = setTimeout(poll, 400);
    return;
  }

  // settled
  clearTimeout(state.poll);
  setBusy(false);
  ui.mini.hidden = true;
  setEngine('ready', 'ready');

  if (status.state === 'error') {
    setView('configure');
    toast(status.error || 'The scan failed.', 'error');
    return;
  }
  if (status.results) {
    showDone(status);
    return;
  }
  await loadClusters();
}

/** Live preview of the photo being processed (GET /thumb, scoped by the engine). */
async function showPreview(name) {
  if (!name || name === state.lastPreview) return;
  state.lastPreview = name;
  ui.previewName.textContent = name;
  try {
    const dataUrl = await bridge.preview(name);
    if (dataUrl && state.lastPreview === name) {
      ui.previewImg.src = dataUrl;
      ui.previewImg.style.animation = 'none';
      void ui.previewImg.offsetWidth;      // restart the crossfade
      ui.previewImg.style.animation = '';
    }
  } catch (error) {
    /* a missing preview is not worth a toast */
  }
}

/* ============================================================ review */

async function loadClusters() {
  try {
    const payload = await bridge.api.clusters();
    state.clusters = payload.clusters || [];
    state.names.clear();
    for (const cluster of state.clusters) {
      if (cluster.name) state.names.set(cluster.id, cluster.name);
    }
    renderClusters();
    setView('review');
    setEngine('ready', 'ready');
    ui.organize.disabled = state.clusters.length === 0;
    if (!state.clusters.length) {
      toast('No faces were found in that folder.', 'warn');
    } else {
      celebrateIfEnabled(26);
    }
  } catch (error) {
    toast(String(error.message || error), 'error');
  }
}

function renderClusters() {
  const query = ui.clusterSearch.value.trim().toLowerCase();
  ui.clusters.replaceChildren();
  let shown = 0;
  state.clusters.forEach((cluster) => {
    const name = (state.names.get(cluster.id) || '').toLowerCase();
    if (query && !name.includes(query)) return;
    const card = buildCard(cluster);
    card.style.animationDelay = `${Math.min(shown, 12) * 45}ms`;
    ui.clusters.append(card);
    shown += 1;
  });
  if (!shown) {
    const empty = document.createElement('p');
    empty.className = 'note';
    empty.style.padding = 'var(--s5) 0';
    empty.textContent = query ? 'No group matches that filter.' : 'No groups yet.';
    ui.clusters.append(empty);
  }
  ui.organize.disabled = state.clusters.length === 0 || state.linking;
  applyLinkState();
}

function buildCard(cluster) {
  const name = state.names.get(cluster.id) || '';
  const wrap = document.createElement('div');
  wrap.className = 'flip';
  wrap.dataset.cluster = String(cluster.id);
  const inner = document.createElement('div');
  inner.className = 'flip-inner';
  wrap.append(inner);

  /* ---- front ---- */
  const front = document.createElement('div');
  front.className = 'flip-face front';
  const top = document.createElement('div');
  top.className = 'card-top';
  // In link mode the whole card is the target, so the title area carries a
  // "First"/"Second" flag telling the user which slot they just filled.
  const pickFlag = document.createElement('span');
  pickFlag.className = 'pick-flag';
  pickFlag.hidden = true;
  const idBox = document.createElement('div');
  const idText = document.createElement('div');
  idText.className = 'card-id';
  // once named, the person IS the card title; the group number stays in the meta
  idText.textContent = name || `Group ${cluster.id + 1}`;
  const meta = document.createElement('div');
  meta.className = 'card-meta';
  meta.textContent = (name ? `Group ${cluster.id + 1} · ` : '')
    + `${plural(cluster.size, 'face')} · ${plural(cluster.photos, 'photo')}`;
  idBox.append(idText, meta);
  // only announce a state that differs from "unnamed" — absence is the default
  const badge = document.createElement('span');
  badge.className = `badge ${name ? (cluster.auto ? 'auto' : 'named') : 'unnamed'}`;
  badge.textContent = name ? (cluster.auto ? 'remembered' : 'named') : '';
  top.append(idBox, badge, pickFlag);

  // Age-invariance note: when the eye region could not be used for most of
  // this group the photos were probably small or blurred, which is exactly
  // when a manual link is the right tool.
  const coverage = typeof cluster.eye_coverage === 'number'
    ? cluster.eye_coverage : null;
  if (coverage !== null && coverage < 0.999) {
    const note = document.createElement('p');
    note.className = 'card-note';
    note.textContent = coverage > 0
      ? `${Math.round(coverage * 100)}% eye-region detail`
      : 'no eye-region detail (small photos)';
    idBox.append(note);
  }

  // In link mode the whole card selects, so clicks must not also open the
  // lightbox or the naming dialog.
  wrap.addEventListener('click', (event) => {
    if (!state.linking) return;
    // Let the card's own buttons keep working.
    if (event.target.closest('button')) return;
    event.preventDefault();
    event.stopPropagation();
    pickForLink(cluster.id);
  });

  const grid = document.createElement('div');
  grid.className = 'thumb-grid';
  for (const face of cluster.faces) {
    // a <figure> inside a <button> is the correct nesting for a labelled image
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'thumb';
    tile.title = `${face.photo} — click to enlarge`;
    if (face.thumb) {
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.src = face.thumb;
      image.alt = `Face from ${face.photo}`;
      image.loading = 'lazy';
      image.decoding = 'async';
      const caption = document.createElement('figcaption');
      caption.textContent = face.photo;
      figure.append(image, caption);
      tile.append(figure);
    }
    tile.addEventListener('click', (event) => {
      event.stopPropagation();
      openLightbox(face);
    });
    grid.append(tile);
  }

  const actions = document.createElement('div');
  actions.className = 'card-actions';
  const nameBtn = document.createElement('button');
  nameBtn.type = 'button';
  nameBtn.className = 'btn primary';
  nameBtn.append(icon('tag'), document.createTextNode(name ? 'Rename' : 'Name'));
  nameBtn.addEventListener('click', () => {
    if (state.linking) { pickForLink(cluster.id); return; }
    openModal(cluster);
  });
  const flipBtn = document.createElement('button');
  flipBtn.type = 'button';
  flipBtn.className = 'btn ghost';
  flipBtn.append(icon('flip'), document.createTextNode('Details'));
  flipBtn.setAttribute('aria-label', `Show details for group ${cluster.id + 1}`);
  flipBtn.addEventListener('click', () => {
    wrap.classList.toggle('is-flipped');
    flipBtn.setAttribute('aria-pressed', wrap.classList.contains('is-flipped') ? 'true' : 'false');
  });
  actions.append(nameBtn, flipBtn);
  front.append(top, grid, actions);

  /* ---- back ---- */
  const back = document.createElement('div');
  back.className = 'flip-face back';
  const backTitle = document.createElement('div');
  backTitle.className = 'card-id';
  backTitle.textContent = name || `Group ${cluster.id + 1}`;
  const details = document.createElement('div');
  details.className = 'detail-list';
  const rows = [
    ['Faces', cluster.size],
    ['Photos', cluster.photos],
    ['Destination', `output/${name || '_unknown'}`],
    ['Status', name ? (cluster.auto ? 'remembered from the name database' : 'named just now') : 'goes to _unknown'],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement('div');
    row.className = 'detail-row';
    const span = document.createElement('span');
    span.textContent = label;
    const strong = document.createElement('b');
    strong.textContent = String(value);
    row.append(span, strong);
    details.append(row);
  }
  const photos = document.createElement('ul');
  photos.className = 'photo-list';
  for (const face of cluster.faces) {
    const item = document.createElement('li');
    item.textContent = face.photo;
    photos.append(item);
  }
  const flipBack = document.createElement('button');
  flipBack.type = 'button';
  flipBack.className = 'btn ghost';
  flipBack.append(icon('refresh'), document.createTextNode('Back'));
  flipBack.addEventListener('click', () => wrap.classList.remove('is-flipped'));
  const backActions = document.createElement('div');
  backActions.className = 'card-actions';
  backActions.append(flipBack);
  back.append(backTitle, details, photos, backActions);

  inner.append(front, back);
  return wrap;
}

/* ================================================== manual age bridge */
/* The safety net: the model cannot always bridge a large age gap, so the
 * user states the ground truth. Two groups become one for this run, and a
 * supplied name is remembered so later scans group them by themselves. */

function clusterById(id) {
  return state.clusters.find((cluster) => cluster.id === id) || null;
}

function clusterLabel(cluster) {
  if (!cluster) return '';
  const name = state.names.get(cluster.id);
  const fallback = `Group ${cluster.id + 1}`;
  if (!name) return fallback;
  // Show the group number too, so two differently-named picks stay
  // distinguishable while linking.
  return `${name} · ${fallback}`;
}

function setLinkMode(on) {
  state.linking = Boolean(on);
  ui.linkToggle.setAttribute('aria-pressed', state.linking ? 'true' : 'false');
  ui.linkToggle.classList.toggle('primary', state.linking);
  ui.linkToggle.classList.toggle('quiet', !state.linking);
  ui.mergeTray.hidden = !state.linking;
  ui.clusters.classList.toggle('is-linking', state.linking);
  ui.clusterSearch.disabled = state.linking;
  if (!state.linking) state.linkPicks = [];
  applyLinkState();
  if (state.linking) {
    ui.linkToggle.blur();
    announce('Link mode on. Select two groups that are the same person.');
  }
}

/** Reflect the current picks in the tray and on the cards. */
function applyLinkState() {
  const slots = [ui.mergeSlotA, ui.mergeSlotB];
  slots.forEach((slot, index) => {
    const id = state.linkPicks[index];
    const cluster = id === undefined ? null : clusterById(id);
    slot.classList.toggle('is-filled', Boolean(cluster));
    const text = slot.querySelector('.merge-slot-empty');
    if (text) {
      text.textContent = cluster
        ? `${clusterLabel(cluster)} · ${cluster.photos} photo(s)`
        : 'Select a group';
    }
  });

  for (const card of ui.clusters.querySelectorAll('.flip')) {
    const id = Number(card.dataset.cluster);
    const picked = state.linkPicks.indexOf(id);
    card.classList.toggle('is-picked', picked > -1);
    // Dim the unpicked ones only once a pick is in progress, so the grid
    // stays fully legible before the user has chosen anything.
    card.classList.toggle('is-dimmed',
      state.linking && state.linkPicks.length === 1 && picked === -1);
    const flag = card.querySelector('.pick-flag');
    if (flag) {
      const active = picked > -1;
      flag.hidden = !active;
      flag.replaceChildren(icon('link'),
        document.createTextNode(picked === 0 ? 'First' : 'Second'));
    }
  }

  ui.mergeConfirm.disabled = state.linkPicks.length !== 2;
  ui.organize.disabled = state.clusters.length === 0 || state.linking;
}

function pickForLink(clusterId) {
  if (!state.linking) return;
  const at = state.linkPicks.indexOf(clusterId);
  if (at > -1) {
    state.linkPicks.splice(at, 1);
  } else if (state.linkPicks.length >= 2) {
    // Two are chosen already: the newest pick replaces the older slot, which
    // is what someone changing their mind expects.
    state.linkPicks = [state.linkPicks[1], clusterId];
  } else {
    state.linkPicks.push(clusterId);
  }
  applyLinkState();
  const count = state.linkPicks.length;
  announce(count === 1
    ? 'First group selected. Now select the second.'
    : (count === 2 ? 'Two groups selected. Link them.' : 'Selection cleared.'));
}

async function confirmMerge() {
  if (state.linkPicks.length !== 2) return;
  const [first, second] = state.linkPicks;
  const name = ui.mergeName.value.trim();
  ui.mergeConfirm.disabled = true;
  try {
    const result = await bridge.api.mergeClusters(first, second, name);
    // Ids were renumbered server-side, so reload rather than patch locally.
    await loadClusters();
    setLinkMode(false);
    ui.mergeName.value = '';
    const remembered = result.remembered
      ? ' — remembered for next time'
      : ' (this run only)';
    toast(`Linked into one group of ${result.size} photo(s)${remembered}`,
      'ok', 'link');
    if (state.celebrate) burst(40);
  } catch (error) {
    toast(String(error.message || error), 'error');
    ui.mergeConfirm.disabled = false;
  }
}

/* ============================================================= modal */

function openModal(cluster) {
  state.current = cluster;
  state.lastFocus = document.activeElement;
  ui.modalThumbs.replaceChildren();
  const faces = (cluster.faces || []).filter((face) => face.thumb).slice(0, 3);
  if (faces.length) {
    for (const face of faces) {
      const image = document.createElement('img');
      image.src = face.thumb;
      image.alt = `Face from ${face.photo}`;
      ui.modalThumbs.append(image);
    }
  } else {
    const placeholder = document.createElement('div');
    placeholder.className = 'placeholder';
    placeholder.append(icon('users'));
    ui.modalThumbs.append(placeholder);
  }
  ui.modalTitle.textContent = (state.names.get(cluster.id))
    ? 'Rename this person' : 'Who is this?';
  ui.modalSub.textContent = `Group ${cluster.id + 1} · ${plural(cluster.size, 'face')} in `
    + `${plural(cluster.photos, 'photo')}`;
  ui.modalName.value = state.names.get(cluster.id) || '';
  ui.modalError.hidden = true;
  ui.modalSuggestions.replaceChildren();
  const suggestions = suggestNames(state.names);
  for (const suggestion of suggestions) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'suggestion';
    chip.textContent = suggestion;
    chip.addEventListener('click', () => { ui.modalName.value = suggestion; });
    ui.modalSuggestions.append(chip);
  }
  ui.modalLayer.hidden = false;
  document.body.style.overflow = 'hidden';
  requestAnimationFrame(() => ui.modalName.focus());
}

function suggestNames(taken) {
  const common = ['Alex', 'Sam', 'Jordan', 'Casey', 'Taylor', 'Morgan'];
  return common.filter((name) => !taken.has(name)).slice(0, 4);
}

function closeModal() {
  ui.modalLayer.hidden = true;
  document.body.style.overflow = '';
  state.current = null;
  if (state.lastFocus && state.lastFocus.focus) state.lastFocus.focus();
}

async function saveModal(name) {
  const cluster = state.current;
  if (!cluster) return;
  try {
    const result = await bridge.api.nameCluster(cluster.id, (name || '').trim());
    if (result.name) {
      state.names.set(cluster.id, result.name);
      toast(`${result.name} — ${result.remembered ? 'remembered for next time' : 'named'}`, 'ok');
      pulseCard(cluster.id);
      celebrateIfEnabled(14);
    } else {
      state.names.delete(cluster.id);
      toast(`Group ${cluster.id + 1} goes to the unknown folder`, 'warn');
    }
    closeModal();
    renderClusters();
  } catch (error) {
    ui.modalError.textContent = String(error.message || error);
    ui.modalError.hidden = false;
  }
}

function pulseCard(clusterId) {
  // match on the data attribute: indices shift when a filter is active, and the
  // visible title becomes the person's name once it is named
  const target = ui.clusters.querySelector(`.flip[data-cluster="${clusterId}"]`);
  if (!target) return;
  target.animate(
    [{ transform: 'scale(1)' }, { transform: 'scale(1.035)' }, { transform: 'scale(1)' }],
    { duration: 420, easing: 'cubic-bezier(.34,1.56,.64,1)' },
  );
}

/* ========================================================= lightbox */

function openLightbox(face) {
  const box = document.createElement('div');
  box.className = 'lightbox';
  const image = document.createElement('img');
  image.src = face.thumb || '';          // face crop first: instant feedback
  image.alt = `Photo ${face.photo}`;
  const caption = document.createElement('div');
  caption.className = 'caption';
  caption.textContent = face.path || face.photo;
  box.append(image, caption);
  box.addEventListener('click', () => box.remove());
  document.addEventListener('keydown', function onKey(event) {
    if (event.key === 'Escape') {
      box.remove();
      document.removeEventListener('keydown', onKey);
    }
  });
  document.body.append(box);

  // then swap in the whole photo (GET /photo, same scoping as /thumb)
  bridge.photo(face.photo).then((dataUrl) => {
    if (dataUrl && box.contains(image)) image.src = dataUrl;
  }).catch(() => { /* the crop is a fine fallback */ });
}

/* ========================================================= organize */

async function runOrganize() {
  if (state.clusters.length === 0) return;
  setBusy(true);
  setEngine('busy', 'sorting');
  ui.mini.hidden = false;
  ui.miniBar.style.width = '0%';
  ui.miniLabel.textContent = 'Sorting photos';
  toast(state.mode === 'move'
    ? 'Moving photos into their folders…'
    : 'Copying photos into their folders…', '', 'folder');
  try {
    await bridge.api.organize(state.mode);
    clearTimeout(state.poll);
    poll();
  } catch (error) {
    setBusy(false);
    setEngine('ready', 'ready');
    ui.mini.hidden = true;
    toast(String(error.message || error), 'error');
  }
}

/* ============================================================== done */

function showDone(status) {
  const results = status.results || {};
  state.outputFolder = results.output_folder || state.outputFolder;
  setView('done');

  const stats = [
    [results.files_placed ?? 0, 'photos placed'],
    [Object.keys(results.folders || {}).length, 'folders created'],
    [results.copies_written ?? 0, 'copies written'],
    [(status.stats && status.stats.faces) || 0, 'faces found'],
  ];
  ui.doneStats.replaceChildren();
  stats.forEach(([value, label]) => {
    // <dl> needs dt/dd, so each cell is a div wrapper holding the pair
    const cell = document.createElement('div');
    const term = document.createElement('dt');
    term.className = 'sr-only';
    term.textContent = label;
    const detail = document.createElement('dd');
    const strong = document.createElement('b');
    strong.textContent = String(value);
    detail.append(strong, document.createTextNode(label));
    cell.append(term, detail);
    ui.doneStats.append(cell);
  });

  ui.folderList.replaceChildren();
  const entries = Object.entries(results.folders || {}).sort((a, b) => b[1] - a[1]);
  for (const [name, count] of entries) {
    const row = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'folder-row';
    button.append(icon('folder'));
    const label = document.createElement('span');
    label.textContent = name;
    const amount = document.createElement('b');
    amount.textContent = String(count);
    button.append(label, amount);
    button.addEventListener('click', () => bridge.openPath(results.output_folder));
    row.append(button);
    ui.folderList.append(row);
  }
  if (!entries.length) {
    const empty = document.createElement('li');
    empty.className = 'note';
    empty.textContent = 'Nothing was placed.';
    ui.folderList.append(empty);
  }

  $('done-sub').textContent = results.mode === 'move'
    ? `${plural(results.files_placed, 'photo')} moved into ${plural(entries.length, 'folder')}.`
    : `${plural(results.files_placed, 'photo')} copied into ${plural(entries.length, 'folder')}.`;

  // No "done" toast: the checkmark, the tally and the folder ledger already
  // state the outcome, and a toast here would sit on top of the action row.
  if (state.celebrate) burst(120);
}

/* ========================================================== confetti */

let confettiFrame = null;

function burst(count = 90) {
  if (reducedMotion) return;
  const canvas = ui.confetti;
  const context = canvas.getContext('2d');
  const ratio = window.devicePixelRatio || 1;
  canvas.width = innerWidth * ratio;
  canvas.height = innerHeight * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  canvas.classList.add('is-on');

  const colors = ['#3b82f6', '#7c5cff', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'];
  const pieces = Array.from({ length: count }, () => ({
    x: Math.random() * innerWidth,
    y: -20 - Math.random() * innerHeight * 0.4,
    w: 6 + Math.random() * 6,
    h: 8 + Math.random() * 8,
    vy: 2 + Math.random() * 3.4,
    vx: -1.2 + Math.random() * 2.4,
    rot: Math.random() * Math.PI,
    vr: -0.12 + Math.random() * 0.24,
    color: colors[(Math.random() * colors.length) | 0],
  }));

  const started = performance.now();
  cancelAnimationFrame(confettiFrame);
  const tick = (now) => {
    const elapsed = now - started;
    context.clearRect(0, 0, innerWidth, innerHeight);
    let alive = false;
    for (const piece of pieces) {
      piece.x += piece.vx;
      piece.y += piece.vy;
      piece.vy += 0.045;              // gentle gravity
      piece.rot += piece.vr;
      if (piece.y < innerHeight + 30) alive = true;
      context.save();
      context.translate(piece.x, piece.y);
      context.rotate(piece.rot);
      context.fillStyle = piece.color;
      context.globalAlpha = Math.max(0, 1 - elapsed / 4200);
      context.fillRect(-piece.w / 2, -piece.h / 2, piece.w, piece.h);
      context.restore();
    }
    if (alive && elapsed < 4200) {
      confettiFrame = requestAnimationFrame(tick);
    } else {
      canvas.classList.remove('is-on');
      context.clearRect(0, 0, innerWidth, innerHeight);
    }
  };
  confettiFrame = requestAnimationFrame(tick);
}

function celebrateIfEnabled(pieces) {
  if (state.celebrate) burst(pieces);
}

/* ===================================================== scroll-to-top */

function updateScrollTop() {
  const offset = ui.stage.scrollTop;
  ui.scrollTop.hidden = offset < 220;
  ui.scrollTop.classList.toggle('is-on', offset >= 220);
}

ui.stage.addEventListener('scroll', updateScrollTop, { passive: true });
ui.scrollTop.addEventListener('click', () => {
  ui.stage.scrollTo({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' });
});

/* ===================================================== drag & drop */

let dragDepth = 0;

function setupDragAndDrop() {
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');

  window.addEventListener('dragenter', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    ui.dragVeil.classList.add('is-on');
    ui.dropzone.classList.add('is-over');
  });
  window.addEventListener('dragover', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) {
      ui.dragVeil.classList.remove('is-on');
      ui.dropzone.classList.remove('is-over');
    }
  });
  window.addEventListener('drop', async (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    ui.dragVeil.classList.remove('is-on');
    ui.dropzone.classList.remove('is-over');
    const file = event.dataTransfer.files[0];
    if (!file) return;
    // Electron >= 32 removed File.path: the preload resolves it via webUtils.
    const path = await bridge.pathForFile(file);
    if (path) {
      ui.input.value = path;
      showChosenFolder(path);
      toast(`Using ${path.split(/[\\/]/).filter(Boolean).pop()}`, 'ok', 'folder');
    }
  });

  ui.dropzone.addEventListener('click', pickInputFolder);
  ui.dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pickInputFolder();
    }
  });
}

async function pickInputFolder() {
  const folder = await bridge.pickFolder('input');
  if (folder) {
    ui.input.value = folder;
    showChosenFolder(folder);
  }
}

/* ============================================================= wiring */

function wire() {
  ui.tolerance.addEventListener('input', () => {
    ui.tolerance.dataset.touched = '1';
    syncRange();
  });
  syncRange();

  for (const field of [ui.minFaces, ui.workers]) {
    field.addEventListener('input', () => { field.dataset.touched = '1'; });
  }
  for (const button of document.querySelectorAll('.seg')) {
    button.addEventListener('click', () => setMode(button.dataset.mode));
  }

  ui.pickInput.addEventListener('click', pickInputFolder);
  ui.pickOutput.addEventListener('click', async () => {
    const folder = await bridge.pickFolder('output');
    if (folder) ui.output.value = folder;
  });

  ui.scan.addEventListener('click', startScan);
  ui.organize.addEventListener('click', runOrganize);
  ui.clusterSearch.addEventListener('input', renderClusters);
  ui.linkToggle.addEventListener('click', () => setLinkMode(!state.linking));
  ui.mergeCancel.addEventListener('click', () => setLinkMode(false));
  ui.mergeConfirm.addEventListener('click', confirmMerge);
  ui.mergeName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); confirmMerge(); }
  });
  ui.autoCelebrate.addEventListener('change', () => {
    state.celebrate = ui.autoCelebrate.checked;
  });

  ui.doneOpen.addEventListener('click', () => bridge.openPath(state.outputFolder));
  ui.doneAgain.addEventListener('click', () => {
    state.clusters = [];
    state.names.clear();
    state.linkPicks = [];
    ui.clusters.replaceChildren();
    setLinkMode(false);
    setView('configure');
    toast('Pick another folder to scan', '', 'refresh');
  });
  ui.doneCelebrate.addEventListener('click', () => burst(140));

  // modal
  ui.modalSave.addEventListener('click', () => saveModal(ui.modalName.value));
  ui.modalSkip.addEventListener('click', () => saveModal(''));
  for (const node of document.querySelectorAll('[data-close-modal]')) {
    node.addEventListener('click', closeModal);
  }
  ui.modalName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); saveModal(ui.modalName.value); }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !ui.modalLayer.hidden) { closeModal(); return; }
    // Escape leaves link mode too, but never while the naming dialog is open.
    if (event.key === 'Escape' && ui.modalLayer.hidden && state.linking) {
      setLinkMode(false);
    }
  });

  // log panel
  ui.logToggle.addEventListener('click', () => {
    ui.log.hidden = !ui.log.hidden;
    ui.logToggle.title = ui.log.hidden ? 'Show engine log' : 'Hide engine log';
  });

  // narrow layout drawer
  ui.drawerToggle.addEventListener('click', () => {
    const open = ui.sidebar.classList.toggle('is-open');
    ui.drawerToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  // engine events
  bridge.onLog(appendLog);
  bridge.onPickInputRequested(pickInputFolder);

  bridge.onReady(async (payload) => {
    setEngine('ready', 'ready');
    applyStatus(payload.status);
    try {
      const status = await bridge.api.status();
      applyStatus(status);
      if (status.state === 'ready') await loadClusters();
      if (status.state === 'done' && status.results) showDone(status);
    } catch (error) {
      /* the scan button reports any real problem */
    }
    if (bridge.smoke) bridge.smoke();
  });
}

function appendLog(line) {
  ui.logPanel.hidden = false;
  ui.log.textContent += `${line}\n`;
  if (ui.log.textContent.length > 24000) {
    ui.log.textContent = ui.log.textContent.slice(-18000);
  }
  ui.log.scrollTop = ui.log.scrollHeight;
}

/* ============================================================== boot */

initTheme();
setupDragAndDrop();
wire();
setView('configure');
setEngine('connecting', 'connecting');