'use strict';

/**
 * FaceSort — Electron main process (the "bridge").
 *
 * Responsibilities, in order:
 *   1. create the window that shows the renderer UI;
 *   2. spawn the Python engine (`backend.exe` in a packaged app, the
 *      PyInstaller output in development);
 *   3. discover the engine's port by parsing the `PORT:<port>` line the
 *      server prints on stdout (see src/api/server.py);
 *   4. wait until `/status` answers, then tell the renderer it may talk to
 *      the engine — the port itself is handed to the renderer by preload.js
 *      through a one-way IPC message;
 *   5. guarantee the engine (and its worker processes) dies with the app.
 *
 * The renderer never gets `require`, `child_process` or the filesystem: it
 * talks to the engine exclusively through the narrow, validated bridge in
 * preload.js.
 */

const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

const IS_PACKAGED = app.isPackaged;
const REPO_ROOT = path.resolve(__dirname, '..');
const PORT_LINE = /^PORT:(\d+)\s*$/;
const READY_TIMEOUT_MS = 180000;   // first scan loads ~190 MB of ONNX weights
const POLL_INTERVAL_MS = 250;

let mainWindow = null;
let backend = null;                // { child, port, logLines: string[] }
let quitting = false;

/* ------------------------------------------------------------------ paths */

/**
 * Where the Python engine lives.
 *
 * Packaged: electron-builder copies `python_build/dist/backend/*` into the
 * app's resources folder (`extraResources` in package.json), so the engine is
 * a sibling of app.asar: `<resources>/backend.exe`.
 *
 * Development: that is the PyInstaller output built by `build.bat` /
 * `build.py backend`, i.e. `python_build/dist/backend/backend[.exe]`.
 */
function backendExecutable() {
  const name = process.platform === 'win32' ? 'backend.exe' : 'backend';
  if (IS_PACKAGED) {
    return path.join(process.resourcesPath, name);
  }
  return path.join(REPO_ROOT, 'python_build', 'dist', 'backend', name);
}

/**
 * Directory the engine is started in. Relative paths in `config.yaml`
 * (e.g. the name DB) resolve against it, so keep it out of the install
 * folder: a real installation must not write into Program Files.
 */
function workingDirectory() {
  return IS_PACKAGED ? app.getPath('userData') : REPO_ROOT;
}

/* ---------------------------------------------------------------- logging */

function log(message) {
  const stamp = new Date().toISOString().slice(11, 19);
  console.log(`[${stamp}] [bridge] ${message}`);
}

function pushBackendLog(line) {
  if (!backend) return;
  backend.logLines.push(line);
  if (backend.logLines.length > 400) backend.logLines.shift();
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:log', line);
  }
}

function fail(message, detail) {
  log(`FATAL: ${message}${detail ? ` — ${detail}` : ''}`);
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:log', `ERROR: ${message}`);
  }
  dialog.showErrorBox('FaceFlow', detail ? `${message}\n\n${detail}` : message);
}

/* -------------------------------------------------------- backend process */

function startBackend() {
  const executable = backendExecutable();
  const cwd = workingDirectory();

  // Print the exact path (and the mode) before touching it: this is the first
  // thing to check when the engine fails to start.
  log(`mode    : ${IS_PACKAGED ? 'packaged' : 'development'}`);
  log(`engine  : ${executable}`);
  log(`cwd     : ${cwd}`);
  if (!fs.existsSync(executable)) {
    fail(
      'The Python engine was not found.',
      `Looked for:\n${executable}\n\n` +
      (IS_PACKAGED
        ? 'The installation looks incomplete — please reinstall the app.'
        : 'Build it first:  python build.py backend   (or run build.bat)')
    );
    return;
  }

  // No model path is passed: the ONNX weights are bundled inside the engine
  // (PyInstaller puts them in models/buffalo_l next to the frozen code) and
  // src.face_model resolves sys._MEIPASS on its own.
  const env = { ...process.env };
  if (IS_PACKAGED) delete env.PYTHONPATH;   // never leak the dev checkout

  backend = { child: null, port: null, logLines: [], exited: false };

  let child;
  try {
    child = spawn(executable, [], {
      env,
      cwd,
      windowsHide: false,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (error) {
    fail('Could not start the Python engine.', String(error));
    return;
  }
  backend.child = child;
  log(`spawned ${executable} (pid ${child.pid})`);

  // --- stdout: the PORT: handshake ------------------------------------
  let buffer = '';
  let portFound = false;
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    buffer += chunk;
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (!portFound) {
        const match = PORT_LINE.exec(trimmed);
        if (match) {
          portFound = true;
          backend.port = Number(match[1]);
          log(`engine reported port ${backend.port}`);
          waitForEngine();
          continue;
        }
      }
      pushBackendLog(trimmed);
    }
  });

  // --- stderr: engine logs --------------------------------------------
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    for (const line of chunk.split(/\r?\n/)) {
      if (line.trim()) pushBackendLog(line.trim());
    }
  });

  child.on('error', (error) => {
    backend.exited = true;
    fail('The Python engine could not be started.', String(error));
  });

  child.on('exit', (code, signal) => {
    backend.exited = true;
    log(`engine exited (code=${code}, signal=${signal})`);
    if (!quitting) {
      pushBackendLog(`The engine stopped unexpectedly (code ${code}).`);
      dialog.showErrorBox(
        'FaceFlow',
        'The Python engine stopped unexpectedly. Please restart the app.'
      );
    }
  });
}

/** Tell the renderer the port plus a first /status payload. Safe to call
 *  repeatedly: preload.js replays the payload to late subscribers and the
 *  renderer treats it as idempotent. */
function announceReady() {
  if (!backend || !backend.port || !mainWindow || mainWindow.isDestroyed()) return;
  fetchStatus()
    .then((status) => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('backend:ready', {
          port: backend.port, status,
        });
      }
    })
    .catch((error) => log(`status fetch failed: ${error}`));
}

/** Poll GET /status until the API answers (the engine's socket is already
 *  listening, so this only waits for uvicorn's startup). */
function waitForEngine() {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  const attempt = () => {
    if (!backend || backend.exited || quitting) return;
    if (Date.now() > deadline) {
      fail('The Python engine did not become ready in time.',
           (backend.logLines || []).slice(-10).join('\n'));
      return;
    }
    const request = http.get(
      { host: '127.0.0.1', port: backend.port, path: '/status', timeout: 3000 },
      (response) => {
        response.resume();
        if (response.statusCode === 200) {
          log('engine is ready');
          announceReady();
          return;
        }
        setTimeout(attempt, POLL_INTERVAL_MS);
      }
    );
    request.on('error', () => setTimeout(attempt, POLL_INTERVAL_MS));
    request.on('timeout', () => request.destroy());
  };
  setTimeout(attempt, POLL_INTERVAL_MS);
}

function fetchStatus() {
  return new Promise((resolve, reject) => {
    if (!backend || !backend.port) return reject(new Error('no port'));
    const request = http.get(
      { host: '127.0.0.1', port: backend.port, path: '/status', timeout: 5000 },
      (response) => {
        let body = '';
        response.setEncoding('utf8');
        response.on('data', (chunk) => { body += chunk; });
        response.on('end', () => {
          try { resolve(JSON.parse(body)); } catch (error) { reject(error); }
        });
      }
    );
    request.on('error', reject);
    request.on('timeout', () => request.destroy(new Error('timeout')));
  });
}

/** Kill the engine and, on Windows, its whole process tree (the M6 worker
 *  pool spawns children; a plain kill() would orphan them). */
function stopBackend() {
  if (!backend || !backend.child || backend.exited) return;
  const { child } = backend;
  const pid = child.pid;
  log(`stopping engine (pid ${pid})`);
  if (process.platform === 'win32' && pid) {
    spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { stdio: 'ignore' });
  } else {
    try { child.kill('SIGTERM'); } catch (error) { /* already gone */ }
  }
  backend.exited = true;
}

/* -------------------------------------------------------------- renderer */

/**
 * Surface renderer problems in the terminal.
 *
 * An Electron window cannot be inspected from outside, so when the UI comes
 * up blank (a CSP violation, a failed file:// load, a GPU crash) the only
 * clue would otherwise be a silent window. Everything the renderer prints —
 * its errors included — is mirrored here, plus a one-line summary of what
 * actually rendered once the page has loaded.
 */
function installRendererDiagnostics(webContents) {
  webContents.on('did-fail-load', (_event, code, description, url) => {
    log(`renderer FAILED to load: ${description} (${code}) ${url}`);
  });

  webContents.on('console-message', (_event, level, message, line, source) => {
    // level: 0 verbose, 1 info, 2 warning, 3 error
    const tag = level >= 3 ? 'ERROR' : level === 2 ? 'WARN' : 'log';
    log(`renderer ${tag}: ${message}  (${source}:${line})`);
  });

  webContents.on('render-process-gone', (_event, details) => {
    log(`renderer process gone: ${details.reason} (exit ${details.exitCode})`);
    if (details.reason === 'crashed') {
      log('hint: try "npx electron . --disable-gpu" if the window stays blank');
    }
  });

  webContents.on('preload-error', (_event, preloadPath, error) => {
    log(`preload FAILED (${preloadPath}): ${error}`);
  });

  webContents.on('unresponsive', () => log('renderer is unresponsive'));
  webContents.on('responsive', () => log('renderer is responsive again'));
  webContents.on('did-finish-load', () => {
    // A cheap assertion that the shell actually mounted. It has caught real
    // load-order and CSP failures, so it stays — but it reports what the
    // current shell contains rather than a removed layout.
    const probe = `(() => {
      const el = (id) => document.getElementById(id);
      const active = document.querySelector('.view.is-active');
      return JSON.stringify({
        title: document.title,
        bridge: typeof window.faceorg,
        shell: !!(el('nav') && el('pane') && el('engine-chip')),
        view: active ? active.dataset.section : null,
        sections: [...document.querySelectorAll('#nav .nav-item')].length,
        theme: document.documentElement.dataset.theme,
        enginePort: (window.faceorg && window.faceorg.ready()) ? 'ready' : 'waiting'
      });
    })()`;
    webContents.executeJavaScript(probe)
      .then((summary) => log(`renderer ready: ${summary}`))
      .catch((error) => log(`renderer probe failed: ${error}`));
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1000,
    minHeight: 680,
    show: false,
    backgroundColor: '#0f1115',
    title: 'FaceFlow',
    icon: path.join(__dirname, 'build', 'icon.ico'),
    // No File/Edit/View/Window/Help bar: it cost a row of window height and
    // every item on it was either already on screen or re-registered as a
    // keyboard shortcut below. Set twice on purpose — the option stops a bar
    // flashing during construction, the call removes it once the window exists.
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      spellcheck: false,
    },
  });

  Menu.setApplicationMenu(null);
  installWindowShortcuts(mainWindow);
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.once('ready-to-show', () => mainWindow.show());
  // A fast engine can be ready before the renderer finished loading, and an
  // IPC message sent to a not-yet-loaded frame would be lost — re-announce.
  mainWindow.webContents.on('did-finish-load', () => announceReady());
  installRendererDiagnostics(mainWindow.webContents);

  // External links open in the real browser, never inside the app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.on('closed', () => { mainWindow = null; });
}

/**
 * The shortcuts that used to live on the removed menu bar.
 *
 * `Menu.setApplicationMenu(null)` takes the accelerators down with the bar, so
 * the three worth keeping are re-registered here:
 *
 *   Ctrl+Shift+I / F12  developer tools. The release README tells people to
 *                       press this when the window comes up blank, so losing
 *                       it would remove the only diagnostic a user has.
 *   Ctrl+O               jump to the screen where a folder is chosen. The old
 *                       menu item did exactly this and nothing more.
 *   Ctrl+R / F5          reload, which is how you recover from a wedged
 *                       renderer without killing the engine.
 *
 * `before-input-event` rather than `globalShortcut`, deliberately: the latter
 * registers with the operating system and would steal these keys from every
 * other application on the machine.
 */
function installWindowShortcuts(win) {
  win.webContents.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown') return;
    const key = String(input.key || '').toLowerCase();
    const ctrl = Boolean(input.control || input.meta);

    if ((ctrl && input.shift && key === 'i') || key === 'f12') {
      event.preventDefault();
      win.webContents.toggleDevTools();
    } else if (ctrl && !input.shift && (key === 'o')) {
      event.preventDefault();
      win.webContents.send('menu:pick-input');
    } else if (key === 'f5' || (ctrl && key === 'r')) {
      event.preventDefault();
      win.webContents.reload();
    }
  });
}

/* ------------------------------------------------------------------ IPC */

const FOLDER_DIALOG_TITLES = {
  output: 'Choose an output folder',
  export: 'Choose a folder for the matching photos',
  gallery: 'Choose a folder for the gallery',
  input: 'Choose an input folder',
};

ipcMain.handle('dialog:pick-folder', async (_event, kind) => {
  const options = {
    title: FOLDER_DIALOG_TITLES[kind] || FOLDER_DIALOG_TITLES.input,
    properties: ['openDirectory', 'createDirectory'],
  };
  const result = mainWindow
    ? await dialog.showOpenDialog(mainWindow, options)
    : { canceled: true, filePaths: [] };
  return result.canceled ? null : result.filePaths[0];
});

/**
 * Reveal a folder in the OS file manager. The renderer only ever passes paths
 * the engine reported (its own output folder), and `shell.openPath` hands the
 * path to the platform's file manager - it never executes anything.
 */
ipcMain.handle('shell:open-path', async (_event, target) => {
  if (typeof target !== 'string' || !target) return 'no path';
  try {
    return await shell.openPath(target);
  } catch (error) {
    return String(error);
  }
});

ipcMain.handle('app:info', () => ({
  version: app.getVersion(),
  packaged: IS_PACKAGED,
  mode: IS_PACKAGED ? 'packaged' : 'development',
  backendPath: backendExecutable(),
  backendCwd: workingDirectory(),
  resourcesPath: IS_PACKAGED ? process.resourcesPath : null,
}));

/* Note: the old Help > About box is gone with the menu bar, and nothing is
   lost by it — Settings > About already shows the version, the engine path, the
   name-database path and an offline callout, which is strictly more than the
   dialog said. Reinstating a dialog that says less would be a downgrade. */

/* ------------------------------------------------------------ lifecycle */

const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    createWindow();
    startBackend();

    /* Automated smoke test: prove the bridge works end to end, print a
     * machine-readable verdict and exit. Used by the build/verification
     * pipeline (FACEORG_SMOKE=1 npm start). */
    if (process.env.FACEORG_SMOKE === '1') {
      ipcMain.on('smoke:ping', async () => {
        try {
          const status = await fetchStatus();
          console.log(`SMOKE_OK ${JSON.stringify({
            packaged: IS_PACKAGED,
            port: backend.port,
            models: status.models,
            state: status.state,
            config: status.config,
          })}`);
        } catch (error) {
          console.log(`SMOKE_FAIL ${error}`);
        }
        setTimeout(() => app.quit(), 250);
      });
      if (mainWindow) {
        mainWindow.webContents.on('did-finish-load', () => {
          mainWindow.webContents.send('smoke:run');
        });
      }
    }

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  quitting = true;
  stopBackend();
});
