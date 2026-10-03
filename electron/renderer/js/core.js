/* ==========================================================================
   FaceFlow — API client, formatting, and the application store
   --------------------------------------------------------------------------
   The engine is a local FastAPI process on 127.0.0.1. Every call goes through
   the preload bridge, so the renderer never learns the port and cannot aim a
   request anywhere else.

   Nothing in this file invents a field. If the engine does not return it, the
   UI shows nothing rather than a plausible-looking zero.
   ========================================================================== */
window.FF = window.FF || {};

(function (FF) {
  'use strict';

  /* =============================================================== errors */

  /**
   * A failure the UI can explain. `hint` is the sentence a non-technical user
   * needs; `detail` is the engine's own wording, kept for the collapsible
   * "technical details" block rather than shown by default.
   */
  class EngineError extends Error {
    constructor(message, hint, detail) {
      super(message);
      this.name = 'EngineError';
      this.hint = hint || '';
      this.detail = detail || '';
    }
  }

  /** The engine is not up yet — the single most common early failure. */
  const NOT_READY = 'The photo engine is still starting. This usually takes a few seconds.';

  /**
   * Run one engine call, normalising every failure into an EngineError.
   *
   * This is `async` on purpose: a promise-returning API must reject, never
   * throw synchronously. When it threw, a caller writing
   * `api.status().catch(...)` had its `.catch` skipped entirely and the
   * exception escaped from whatever function made the call.
   */
  async function call(fn) {
    if (!window.faceorg || !window.faceorg.ready()) {
      throw new EngineError('Engine not ready', NOT_READY);
    }
    try {
      return await fn();
    } catch (error) {
      const message = String((error && error.message) || error);
      if (/still starting up/i.test(message)) {
        throw new EngineError('Engine not ready', NOT_READY, message);
      }
      if (/No scan results yet/i.test(message)) {
        throw new EngineError('No scan yet',
          'Run a scan first — there is nothing to group until then.', message);
      }
      if (/ECONNREFUSED|Failed to fetch|terminated/i.test(message)) {
        throw new EngineError('Engine stopped responding',
          'The local photo engine stopped. Restart FaceFlow and try again.', message);
      }
      throw new EngineError(message, '', message);
    }
  }

  /**
   * The engine endpoints, one method each. Named after the route so the
   * contract is greppable from either end.
   */
  const api = {
    status:      () => call(() => window.faceorg.api.status()),
    scan:        (body) => call(() => window.faceorg.api.scan(body)),
    clusters:    () => call(() => window.faceorg.api.clusters()),
    nameCluster: (id, name) => call(() => window.faceorg.api.nameCluster(id, name)),
    mergeClusters: (a, b, name) => call(() => window.faceorg.api.mergeClusters(a, b, name)),
    peopleList:  () => call(() => window.faceorg.api.peopleList()),
    intersection:(names) => call(() => window.faceorg.api.intersection(names)),
    coOccurrence:() => call(() => window.faceorg.api.coOccurrence()),
    exportIntersection: (names, folder) =>
      call(() => window.faceorg.api.exportIntersection(names, folder)),
    exportGallery:(body) => call(() => window.faceorg.api.exportGallery(body)),
    galleryStatus:() => call(() => window.faceorg.api.galleryStatus()),
    organize:    (mode) => call(() => window.faceorg.api.organize(mode)),
    preview:     async (name) => (window.faceorg.ready() ? window.faceorg.preview(name) : null),
    photo:       async (name) => (window.faceorg.ready() ? window.faceorg.photo(name) : null),
    pickFolder:  (kind) => window.faceorg.pickFolder(kind),
    openPath:    (target) => window.faceorg.openPath(target),
    appInfo:     () => window.faceorg.appInfo(),
    pathForFile: (file) => window.faceorg.pathForFile(file),
    platform:    () => window.faceorg.platform,
  };

  /* ============================================================ formatting */

  /** "1 photo" / "4 photos". Irregulars are passed explicitly, not guessed. */
  function plural(count, word, irregularPlural) {
    const noun = count === 1 ? word : (irregularPlural || `${word}s`);
    return `${formatCount(count)} ${noun}`;
  }

  const formatCount = (value) =>
    (typeof value === 'number' && Number.isFinite(value) ? value : 0).toLocaleString('en-US');

  function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let value = n / 1024;
    let index = 0;
    while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
    return `${value.toFixed(value < 10 ? 1 : 0)} ${units[index]}`;
  }

  /** Relative for recent, absolute once "2 days ago" stops being useful. */
  function formatWhen(value) {
    if (!value) return '';
    const then = typeof value === 'number' ? new Date(value) : new Date(value);
    if (Number.isNaN(then.getTime())) return '';
    const seconds = Math.round((Date.now() - then.getTime()) / 1000);
    if (seconds < 45) return 'just now';
    if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
    if (seconds < 604800) return `${Math.round(seconds / 86400)} d ago`;
    return then.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  function formatDuration(seconds) {
    const n = Number(seconds) || 0;
    if (n < 60) return `${n.toFixed(n < 10 ? 1 : 0)}s`;
    const minutes = Math.floor(n / 60);
    const rest = Math.round(n % 60);
    if (minutes < 60) return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
    return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  }

  const basename = (path) => {
    const parts = String(path || '').split(/[\\/]/).filter(Boolean);
    return parts[parts.length - 1] || String(path || '');
  };

  const separator = () => (api.platform() === 'win32' ? '\\' : '/');

  /** Join a chosen parent with a folder name, using the right separator. */
  const joinPath = (parent, child) =>
    `${String(parent).replace(/[\\/]+$/, '')}${separator()}${child}`;

  /* ================================================================= store */

  /**
   * One observable object. Views subscribe to keys and re-render on change;
   * nothing polls the DOM for state.
   */
  function createStore(initial) {
    const state = { ...initial };
    const subscribers = new Set();
    let queued = false;
    let pendingKeys = new Set();

    function notify(keys) {
      // Accumulate, do not replace: several sets can land in the same
      // microtask, and each subscriber needs to see every key that changed.
      for (const key of keys) pendingKeys.add(key);
      if (queued) return;
      queued = true;
      queueMicrotask(() => {
        queued = false;
        const changed = [...pendingKeys];
        pendingKeys = new Set();
        for (const subscriber of subscribers) {
          try { subscriber(state, changed); } catch (error) { console.error(error); }
        }
      });
    }

    return {
      get state() { return state; },
      set(patch) {
        const keys = [];
        for (const key of Object.keys(patch)) {
          if (state[key] !== patch[key]) keys.push(key);
        }
        Object.assign(state, patch);
        if (keys.length) notify(keys);
      },
      subscribe(fn) {
        subscribers.add(fn);
        return () => subscribers.delete(fn);
      },
    };
  }

  FF.api = api;
  FF.EngineError = EngineError;
  FF.fmt = {
    plural, count: formatCount, bytes: formatBytes, when: formatWhen,
    duration: formatDuration, basename, separator, joinPath,
  };
  FF.createStore = createStore;
})(window.FF);