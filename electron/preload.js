'use strict';

/**
 * Preload script — the only bridge between the renderer and Node.
 *
 * The renderer runs with `contextIsolation: true` and `nodeIntegration:
 * false`, so it can neither read files nor spawn processes.  Everything it
 * is allowed to do is enumerated here:
 *
 *   - `onReady(cb)`      main.js finished the engine handshake
 *   - `onLog(cb)`        live engine log lines (sidebar log panel)
 *   - `api.*`            the five REST endpoints of the local engine
 *   - `pickFolder(kind)` native folder picker
 *   - `info`             app/engine locations (shown in the About box)
 *
 * REST calls go out from this privileged context (Node's global `fetch`), so
 * the renderer never needs to know the port, and requests are limited to a
 * hard-coded 127.0.0.1 base URL — there is no way to aim them elsewhere.
 */

const { contextBridge, ipcRenderer, webUtils } = require('electron');

let baseUrl = null;
let lastReady = null;            // replayed to late subscribers (see onReady)
const listeners = { ready: [], log: [], pickInput: [] };

function emit(kind, payload) {
  for (const callback of listeners[kind] || []) {
    try {
      callback(payload);
    } catch (error) {
      console.error('listener failed', error);
    }
  }
}

ipcRenderer.on('backend:ready', (_event, payload) => {
  baseUrl = `http://127.0.0.1:${payload.port}`;
  lastReady = payload;
  emit('ready', payload);
});
ipcRenderer.on('backend:log', (_event, line) => emit('log', line));
ipcRenderer.on('menu:pick-input', () => emit('pickInput', true));

async function call(pathname, options = {}) {
  if (!baseUrl) {
    throw new Error('The engine is still starting up — try again in a moment.');
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeout || 30000);
  try {
    const response = await fetch(`${baseUrl}${pathname}`, {
      method: options.method || 'GET',
      headers: options.body ? { 'Content-Type': 'application/json' } : {},
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (error) {
        throw new Error(`Unexpected reply from the engine: ${text.slice(0, 200)}`);
      }
    }
    if (!response.ok) {
      const detail = (payload && payload.detail) || `HTTP ${response.status}`;
      throw new Error(detail);
    }
    return payload;
  } finally {
    clearTimeout(timer);
  }
}

contextBridge.exposeInMainWorld('faceorg', {
  ready: () => baseUrl !== null,
  /**
   * Subscribing replays the ready event if it already happened. The engine
   * can be ready in a few hundred milliseconds — often before the renderer's
   * own script has finished loading — and an un-replayed event would leave
   * the UI waiting forever for a port it never receives.
   */
  onReady(callback) {
    listeners.ready.push(callback);
    if (lastReady) {
      try { callback(lastReady); } catch (error) { console.error(error); }
    }
  },
  onLog(callback) { listeners.log.push(callback); },
  onPickInputRequested(callback) { listeners.pickInput.push(callback); },

  api: {
    status: () => call('/status'),
    scan: (body) => call('/scan', { method: 'POST', body, timeout: 60000 }),
    clusters: () => call('/clusters', { timeout: 120000 }),
    nameCluster: (clusterId, name) =>
      call('/name_cluster', { method: 'POST', body: { cluster_id: clusterId, name } }),
    /**
     * Link two groups as the same person (typically a childhood group and an
     * adult one). Supplying a name also remembers the link, so later scans
     * group them automatically.
     */
    mergeClusters: (clusterA, clusterB, name) =>
      call('/merge_clusters', {
        method: 'POST',
        body: { cluster_a: clusterA, cluster_b: clusterB, name: name || '' },
        timeout: 60000,
      }),
    organize: (mode) =>
      call('/organize', { method: 'POST', body: mode ? { mode } : {}, timeout: 300000 }),
  },

  /**
   * Downscaled JPEG of the photo currently being processed, as a data URL.
   * Binary stays in this privileged context: the renderer only ever sees
   * `data:` URLs, so its CSP needs no extra source and no blob lifecycle.
   */
  preview: async (name) => {
    if (!baseUrl || !name) return null;
    const response = await fetch(
      `${baseUrl}/thumb?name=${encodeURIComponent(name)}`);
    if (!response.ok) return null;
    const buffer = Buffer.from(await response.arrayBuffer());
    return `data:image/jpeg;base64,${buffer.toString('base64')}`;
  },

  /** Whole photo (downscaled JPEG, data URL) for the review lightbox. */
  photo: async (name) => {
    if (!baseUrl || !name) return null;
    const response = await fetch(
      `${baseUrl}/photo?name=${encodeURIComponent(name)}`);
    if (!response.ok) return null;
    const buffer = Buffer.from(await response.arrayBuffer());
    return `data:image/jpeg;base64,${buffer.toString('base64')}`;
  },

  pickFolder: (kind) => ipcRenderer.invoke('dialog:pick-folder', kind),
  appInfo: () => ipcRenderer.invoke('app:info'),

  /**
   * Absolute path of a dropped File/Item. Electron removed `File.path`, so
   * `webUtils.getPathForFile` is the supported way to resolve it.
   */
  pathForFile: (file) => {
    try { return webUtils.getPathForFile(file); } catch (error) { return null; }
  },

  /** Reveal a folder in the OS file manager (output folder, folder rows). */
  openPath: (target) => ipcRenderer.invoke('shell:open-path', target),

  /* Used only by the automated smoke test (FACEORG_SMOKE=1). */
  smoke: process.env.FACEORG_SMOKE === '1'
    ? () => ipcRenderer.send('smoke:ping')
    : null,
});
