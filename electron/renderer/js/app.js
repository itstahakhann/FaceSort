/* ==========================================================================
   FaceFlow — application shell
   Boots the engine handshake, owns navigation, theme, the activity feed and
   the polling that keeps the shell honest while a job runs.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon, debounce } = FF.dom;
  const { api, fmt } = FF;
  const { toast, callout } = FF.ui;

  const THEME_KEY = 'facesort.theme';
  const SECTIONS = [
    { id: 'overview',     label: 'Overview',          icon: 'home',     title: 'Overview' },
    { id: 'people',       label: 'People',            icon: 'users',    title: 'People' },
    { id: 'photos',       label: 'Photos',            icon: 'image',    title: 'Photos' },
    { id: 'relationships', label: 'Relationships',    icon: 'link',     title: 'Relationships' },
    { id: 'gallery',      label: 'Gallery Export',    icon: 'camera',   title: 'Gallery Export' },
  ];

  const store = FF.createStore({
    section: 'overview',
    engine: 'connecting',      // connecting | ready | busy | error
    engineLabel: 'starting',
    theme: 'dark',
    online: navigator.onLine,
    clusters: 0,
    named: 0,
    unknown: 0,
    activity: [],
  });
  FF.store = store;

  /* ============================================================== activity */

  function logActivity(text, kind) {
    const entry = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                    text, kind: kind || '', at: Date.now() };
    const next = [entry, ...store.state.activity].slice(0, 24);
    store.set({ activity: next });
  }
  FF.logActivity = logActivity;

  function renderActivity() {
    const list = $('#activity-list');
    const items = store.state.activity;
    if (!items.length) {
      mount(list, h('p', { class: 'activity-empty',
        text: 'Scan, name and export — your actions appear here.' }));
      return;
    }
    mount(list, items.map((entry) => h('div', { class: `activity-item ${entry.kind ? `is-${entry.kind}` : ''}` },
      h('span', { class: 'dot', 'aria-hidden': 'true' }),
      h('div', null,
        h('span', { text: entry.text }),
        h('time', { dateTime: new Date(entry.at).toISOString(), text: fmt.when(entry.at) })))));
  }

  /* ================================================================ chrome */

  function renderNav() {
    const nav = $('#nav');
    mount(nav, SECTIONS.map((section) => h('button', {
      class: 'nav-item',
      type: 'button',
      'aria-current': store.state.section === section.id ? 'page' : null,
      onclick: () => go(section.id),
    },
      icon(section.icon),
      h('span', { text: section.label }),
      countFor(section.id))));
  }

  function countFor(sectionId) {
    const { clusters, named } = store.state;
    if (sectionId === 'photos' && clusters) return h('span', { class: 'nav-count', text: fmt.count(clusters) });
    if (sectionId === 'people' && named) return h('span', { class: 'nav-count', text: fmt.count(named) });
    // Relationships has no count of its own — a number there would be read as
    // "this many relationships", which the engine does not report.
    return null;
  }

  function setEngine(state, label) {
    store.set({ engine: state, engineLabel: label || state });
  }

  function renderChrome() {
    const chip = $('#engine-chip');
    chip.dataset.state = store.state.engine;
    $('#engine-label').textContent = store.state.engineLabel;

    // The privacy chip doubles as a network indicator, because "offline" is
    // the product's central claim and a user should be able to see it hold.
    const chipNode = $('#offline-chip');
    const online = store.state.online;
    chipNode.classList.toggle('is-warning', !online && store.state.engine === 'ready');
    chipNode.querySelector('span:last-child').textContent = online
      ? 'Offline · local only'
      : 'No network · still working';

    const section = SECTIONS.find((item) => item.id === store.state.section);
    $('#page-title').textContent = section ? section.title : 'FaceFlow';
    $('#nav-settings').setAttribute('aria-current',
      store.state.section === 'settings' ? 'page' : 'false');

    const iconUse = $('#theme-icon use');
    iconUse.setAttribute('href', store.state.theme === 'dark' ? '#i-sun' : '#i-moon');
    const toggle = $('#theme-toggle');
    const nextLabel = store.state.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
    toggle.setAttribute('aria-label', nextLabel);
    toggle.title = nextLabel;
  }

  function announce(text) { $('#live').textContent = text; }
  FF.announce = announce;

  /* ================================================================= theme */

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch (error) { /* private mode */ }
    store.set({ theme });
  }

  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch (error) { /* ignore */ }
    const preferred = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    applyTheme(stored || preferred);
    $('#theme-toggle').addEventListener('click', () => {
      applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    });
  }

  /* =============================================================== routing
     Views are functions returning a node and optionally a teardown. The shell
     owns the lifecycle so views never have to think about it. */

  /* Views register themselves into FF.views as their script loads; app.js is
     the last script in index.html, so the registry is already populated here. */
  const views = FF.views || (FF.views = {});
  let teardown = null;

  function go(section) {
    const view = views[section] || views.overview;
    if (!view) {
      console.error(`no view registered for "${section}"`);
      return;
    }
    if (typeof teardown === 'function') { try { teardown(); } catch (error) { console.error(error); } }
    teardown = null;

    store.set({ section });
    renderNav();
    renderChrome();

    const result = view(store.state);
    const node = result && result.node ? result.node : result;
    teardown = result && result.teardown ? result.teardown : null;

    const pane = $('#pane-inner');
    pane.replaceChildren();
    const wrap = h('div', { class: 'view is-active', dataset: { section } }, node);
    pane.append(wrap);
    $('#pane').scrollTop = 0;
    announce(`${(SECTIONS.find((s) => s.id === section) || {}).title || section} opened`);
    if (window.faceorg && window.faceorg.onPickInputRequested) {
      window.faceorg.onPickInputRequested(() => go('scan'));
    }
  }
  FF.go = go;

  /** Re-render the current section in place. */
  FF.rerender = () => go(store.state.section);

  /* ============================================================== polling
     One chain for the whole app, and only while something is actually
     running. Leaving a view unmounts its listeners, but the shell's poll
     must not outlive the job it is watching. */

  let pollTimer = null;
  let poller = null;

  function stopPolling() {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    poller = null;
  }

  /**
   * `handler(status)` runs on each tick. Polling stops as soon as the status
   * reports neither scanning nor organizing, so an idle app makes no requests
   * at all.
   */
  function watch(handler, interval) {
    poller = handler;
    const tick = async () => {
      if (poller !== handler) return;
      let status;
      try {
        status = await api.status();
      } catch (error) {
        setEngine('error', 'unavailable');
        stopPolling();
        return;
      }
      applyStatusToShell(status);
      if (poller) await poller(status);
      const busy = status.scanning || status.organizing || (status.gallery && status.gallery.building);
      if (busy && poller === handler) {
        pollTimer = setTimeout(tick, interval || 420);
      } else {
        stopPolling();
      }
    };
    if (!pollTimer) tick();
  }
  FF.watch = watch;
  FF.stopWatching = stopPolling;

  function applyStatusToShell(status) {
    const clusters = status.clusters || 0;
    const named = status.named || 0;
    const unknown = status.unknown || 0;
    const changed = clusters !== store.state.clusters || named !== store.state.named;
    store.set({ clusters, named, unknown });

    if (status.scanning) setEngine('busy', 'scanning');
    else if (status.organizing) setEngine('busy', 'sorting');
    else if (status.state === 'error') setEngine('error', 'error');
    else setEngine('ready', 'ready');

    if (changed) {
      renderNav();
      // Only the counts in the sidebar change while a job runs; re-rendering
      // the whole view on every tick would fight the user mid-scroll.
      if (!status.scanning && !status.organizing) {
        const section = SECTIONS.find((item) => item.id === store.state.section);
        if (section && ['overview', 'photos', 'people'].includes(section.id)) FF.rerender();
      }
    }
  }
  FF.applyStatusToShell = applyStatusToShell;

  /* ============================================================ drag/drop
     Dropping a folder anywhere in the window selects it as the input folder
     and jumps to the scan view. */

  function initDragAndDrop() {
    const veil = h('div', { class: 'drop-veil', id: 'drop-veil', hidden: true },
      h('div', { class: 'drop-veil-inner' },
        icon('folder-open', 'icon-lg'),
        h('strong', { text: 'Drop to use this folder' })));
    document.body.append(veil);
    let depth = 0;

    const hasFiles = (event) =>
      Array.from((event.dataTransfer && event.dataTransfer.types) || []).includes('Files');

    window.addEventListener('dragenter', (event) => {
      if (!hasFiles(event)) return;
      depth += 1;
      veil.hidden = false;
    });
    window.addEventListener('dragover', (event) => event.preventDefault());
    window.addEventListener('dragleave', () => {
      depth = Math.max(0, depth - 1);
      if (!depth) veil.hidden = true;
    });
    window.addEventListener('drop', async (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth = 0;
      veil.hidden = true;
      const file = event.dataTransfer.files[0];
      if (!file) return;
      const path = await api.pathForFile(file);
      if (path) {
        FF.store.set({ droppedInputFolder: path });
        logActivity('Folder dropped', 'accent');
        go('scan');
        toast('Folder ready to scan', 'success');
      }
    });
  }

  /* =================================================================== boot */

  async function boot() {
    initTheme();
    initDragAndDrop();
    renderActivity();
    renderNav();
    renderChrome();

    store.subscribe((_state, keys) => {
      if (keys.some((key) => ['engine', 'engineLabel', 'theme', 'online', 'section'].includes(key))) {
        renderChrome();
      }
      if (keys.includes('activity')) renderActivity();
      if (keys.some((key) => ['clusters', 'named', 'unknown', 'section'].includes(key))) renderNav();
    });

    $('#activity-clear').addEventListener('click', () => store.set({ activity: [] }));

    window.addEventListener('online', () => { store.set({ online: true }); logActivity('Network restored — the app does not use it'); });
    window.addEventListener('offline', () => { store.set({ online: false }); });

    // Global shortcuts. Only when nothing is typing, so a space bar in a text
    // field never navigates away from what the user is doing.
    document.addEventListener('keydown', (event) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      const index = Number(event.key) - 1;
      if (Number.isInteger(index) && index >= 0 && index < SECTIONS.length) {
        event.preventDefault();
        go(SECTIONS[index].id);
      }
    });

    setEngine('connecting', 'starting');
    go('overview');

    window.faceorg.onReady(async () => {
      setEngine('ready', 'ready');
      try {
        const [status, info] = await Promise.all([api.status(), api.appInfo()]);
        FF.store.set({ appInfo: info });
        applyStatusToShell(status);
        const settings = (status.config) || {};
        FF.store.set({
          config: {
            inputFolder: settings.input_folder || '',
            outputFolder: settings.output_folder || '',
            tolerance: settings.tolerance ?? 0.5,
            minFaces: settings.min_faces_per_cluster ?? 2,
            mode: settings.mode === 'move' ? 'move' : 'copy',
            workers: settings.workers ?? 0,
            unknownFolder: settings.unknown_folder || '_unknown',
            namesDb: settings.names_db || '',
            inputExists: Boolean(settings.input_exists),
            outputExists: Boolean(settings.output_exists),
          },
          stats: status.stats || {},
          lastResults: status.results || null,
        });
        logActivity('Engine ready', 'success');
      } catch (error) {
        setEngine('error', 'unavailable');
        logActivity('Engine handshake failed', 'error');
      }
    });

    /* The engine streams its own log. Most of it is uvicorn boot chatter, and
       some of it contains local paths — neither belongs in a sidebar meant to
       reassure. Only genuine failures are surfaced, with the timestamp and
       logger name stripped. */
    const LOG_NOISE = /uvicorn|Application startup|Application shutdown|Waiting for application|Started server process|Shutting down|Started reloader|WASMS|^\s*$/i;
    window.faceorg.onLog((line) => {
      const text = String(line || '');
      if (LOG_NOISE.test(text)) return;
      const isFailure = /\berror\b|\bfailed\b|traceback|exception/i.test(text)
        && !/0 errors|no errors/i.test(text);
      if (!isFailure) return;
      // Drop the leading "2026-10-03 11:35:15.943 INFO uvicorn.error: " part so
      // the sentence reads as prose rather than as a log record.
      const clean = text.replace(/^\s*[\d-]{10}\s[\d:.,]+\s+(INFO|WARNING|ERROR|DEBUG)\s+[\w.]+:\s*/i, '')
        .split('\n')[0].trim();
      if (!clean) return;
      logActivity(clean.slice(0, 72), 'error');
    });
  }

  /** Shared settings write. There is no POST /config on the engine: the
   *  effective configuration is the body of the next /scan, so settings are
   *  held in the renderer and applied when a scan starts. */
  FF.readSettings = () => {
    const config = store.state.config || {};
    return {
      input_folder: config.inputFolder,
      output_folder: config.outputFolder,
      tolerance: Number(config.tolerance) || 0.5,
      min_faces_per_cluster: Number(config.minFaces) || 2,
      mode: config.mode === 'move' ? 'move' : 'copy',
      workers: Number(config.workers) || 0,
      use_db: true,
    };
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  FF.SECTIONS = SECTIONS;
  FF.setEngine = setEngine;
})(window.FF);