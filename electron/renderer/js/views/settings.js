/* ==========================================================================
   FaceFlow — Settings
   Every control here maps to a key the engine already reads. There is no
   POST /config on the engine: configuration is the body of the next scan, so
   changes are held locally and applied when a scan starts. The page says so
   rather than implying a save that does not exist.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, icon } = FF.dom;
  const { api, fmt } = FF;
  const { toast, callout } = FF.ui;

  FF.views = FF.views || {};

  FF.views.settings = function (state) {
    const host = h('div', { class: 'view-body' });
    let disposed = false;
    let info = null;

    /* ------------------------------------------------------------ helpers */

    const config = () => FF.store.state.config || {};

    function update(key, value) {
      FF.store.set({ config: { ...config(), [key]: value } });
    }

    function row(label, control, note) {
      return h('div', { class: 'setting-row' },
        h('div', { class: 'setting-label' },
          h('div', { class: 'setting-name', text: label }),
          note ? h('div', { class: 'setting-note', text: note }) : null),
        h('div', { class: 'setting-control' }, control));
    }

    function textField(key, label, note, kind) {
      const input = h('input', {
        class: 'input input-mono', type: 'text', id: `ff-set-${key}`,
        value: config()[key] || '', spellcheck: 'false', autocomplete: 'off',
        placeholder: 'Not set',
      });
      input.addEventListener('input', () => update(key, input.value.trim()));
      return row(label,
        h('div', { class: 'input-row' },
          input,
          h('button', {
            class: 'btn btn-icon', type: 'button',
            'aria-label': `Choose ${label.toLowerCase()}`,
            onclick: async () => {
              const folder = await api.pickFolder(kind);
              if (!folder) return;
              update(key, folder);
              render();
            },
          }, icon('folder-open'))),
        note);
    }

    function numberField(key, label, min, max, step, note) {
      const input = h('input', {
        class: 'input', type: 'number', id: `ff-set-${key}`,
        min, max, step, value: config()[key] ?? min,
      });
      input.addEventListener('input', () => update(key, Number(input.value)));
      return row(label, input, note);
    }

    function rangeField(key, label, min, max, step, note) {
      const input = h('input', {
        type: 'range', id: `ff-set-${key}`, min, max, step, value: config()[key] ?? 0.5,
      });
      const readout = h('output', { class: 't-num field-value',
        text: Number(input.value).toFixed(2) });
      const paint = () => {
        const pct = ((input.value - min) / (max - min)) * 100;
        input.style.setProperty('--fill', `${pct}%`);
        readout.textContent = Number(input.value).toFixed(2);
      };
      input.addEventListener('input', () => {
        update(key, Number(input.value));
        paint();
      });
      paint();
      return row(label, h('div', { class: 'range-wrap' }, input, readout), note);
    }

    /* -------------------------------------------------------------- render */

    function render() {
      if (disposed) return;
      const current = config();

      mount(host,
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'Settings' }),
          h('p', { class: 'page-sub',
            text: 'These values are sent to the engine with your next scan.' })),

        /* ------------------------------------------------------- scanning */
        h('section', { class: 'section' },
          h('h2', { class: 'section-title', text: 'Scanning' }),
          h('div', { class: 'settings-list' },
            textField('inputFolder', 'Photo folder', 'The folder FaceFlow reads. Never modified.', 'input'),
            textField('outputFolder', 'Destination folder',
              'Sorted folders are created inside it.', 'output'),
            numberField('minFaces', 'Minimum faces per group',
              1, 50, 1,
              'Groups with fewer faces than this are treated as noise. Raise it to reduce '
              + 'stray faces, lower it to keep small groups.'),
            numberField('workers', 'Worker processes', 0, 16, 1,
              '0 chooses a size based on your CPU. More workers finish sooner but use more memory.'))),

        /* --------------------------------------------------- organization */
        h('section', { class: 'section' },
          h('h2', { class: 'section-title', text: 'Organization' }),
          h('div', { class: 'settings-list' },
            rangeField('tolerance', 'Match strictness', 0.2, 1, 0.05,
              'How similar two faces must be to land in the same group. Higher merges more; '
              + 'lower splits more.'),
            row('When sorting',
              h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'Copy or move' },
                ...['copy', 'move'].map((mode) => h('button', {
                  type: 'button', role: 'radio',
                  'aria-checked': String(current.mode === mode),
                  class: current.mode === mode ? 'is-active' : '',
                  onclick: () => { update('mode', mode); render(); },
                }, icon(mode === 'copy' ? 'copy' : 'move', 'icon-sm'),
                   mode === 'copy' ? 'Copy photos' : 'Move photos'))),
              'Copy leaves your originals untouched. Move removes them from the source folder.'),
            row('Folder for unnamed groups',
              h('code', { class: 'input-mono', text: current.unknownFolder || '_unknown' }),
              'Groups you skip go here, so nothing is ever lost.'))),

        /* ----------------------------------------------------- appearance */
        h('section', { class: 'section' },
          h('h2', { class: 'section-title', text: 'Appearance' }),
          h('div', { class: 'settings-list' },
            row('Theme',
              h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'Theme' },
                ...['dark', 'light'].map((theme) => h('button', {
                  type: 'button', role: 'radio',
                  'aria-checked': String(state.theme === theme),
                  class: state.theme === theme ? 'is-active' : '',
                  onclick: () => {
                    document.documentElement.dataset.theme = theme;
                    try { localStorage.setItem('facesort.theme', theme); } catch (error) { /* ignore */ }
                    FF.store.set({ theme });
                    render();
                  },
                }, icon(theme === 'dark' ? 'moon' : 'sun', 'icon-sm'),
                   theme === 'dark' ? 'Dark' : 'Light'))),
              'Follows your system setting until you choose one here.'))),

        /* ------------------------------------------------------ about */
        h('section', { class: 'section' },
          h('h2', { class: 'section-title', text: 'About' }),
          h('div', { class: 'settings-list' },
            row('Version', h('span', { class: 't-mono', text: info ? info.version : '—' })),
            row('Engine',
              h('span', { class: 't-mono truncate', style: { maxWidth: '380px' },
                text: info ? info.backendPath : '—' }),
              'The local process that detects faces. It never leaves this machine.'),
            row('Name database',
              h('span', { class: 't-mono truncate', style: { maxWidth: '380px' },
                text: current.namesDb || '—' }),
              'Remembers the names you give your groups.'),
            row('Network',
              callout({
                kind: 'accent',
                icon: 'shield',
                message: 'FaceFlow makes no network requests. There is no account, no '
                      + 'analytics and no upload. The only process it talks to is the local '
                      + 'engine above.',
              })))));
    }

    api.appInfo().then((payload) => { info = payload; if (!disposed) render(); }).catch(() => {});
    render();
    return { node: host, teardown: () => { disposed = true; } };
  };
})(window.FF);