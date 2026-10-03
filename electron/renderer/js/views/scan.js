/* ==========================================================================
   FaceFlow — Scan
   Two states in one view: the configuration form, and the live scan. The
   transition between them is the only place the app takes control away, so
   it is also where the progress reporting has to be trustworthy.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon } = FF.dom;
  const { api, fmt } = FF;
  const { toast, callout, errorCallout, emptyState } = FF.ui;

  FF.views = FF.views || {};

  /* --------------------------------------------------------------- inputs */

  function folderField(label, key, kind, hint) {
    const input = h('input', {
      class: 'input input-mono',
      type: 'text',
      id: `ff-${key}`,
      spellcheck: 'false',
      autocomplete: 'off',
      placeholder: 'No folder chosen',
      value: (FF.store.state.config || {})[key] || '',
    });
    input.addEventListener('input', () => {
      const config = { ...(FF.store.state.config || {}) };
      config[key] = input.value.trim();
      FF.store.set({ config });
    });
    const browse = h('button', {
      class: 'btn btn-icon',
      type: 'button',
      'aria-label': `Choose ${label.toLowerCase()} folder`,
      title: `Choose ${label.toLowerCase()} folder`,
      onclick: async () => {
        const folder = await api.pickFolder(kind);
        if (!folder) return;
        input.value = folder;
        input.dispatchEvent(new Event('input'));
      },
    }, icon('folder-open'));

    return h('div', { class: 'field' },
      h('label', { class: 'label', for: `ff-${key}`, text: label }),
      h('div', { class: 'input-row' }, input, browse),
      hint ? h('p', { class: 'field-note', text: hint }) : null);
  }

  function rangeField(label, key, min, max, step, format) {
    const config = () => FF.store.state.config || {};
    const value = config()[key];
    const input = h('input', {
      type: 'range', id: `ff-${key}`, min, max, step, value,
      'aria-describedby': `ff-${key}-value`,
    });
    const readout = h('output', { class: 't-num field-value', id: `ff-${key}-value`,
      text: format(value) });
    const paint = () => {
      const pct = ((input.value - min) / (max - min)) * 100;
      input.style.setProperty('--fill', `${pct}%`);
      readout.textContent = format(input.value);
    };
    input.addEventListener('input', () => {
      const next = { ...(FF.store.state.config || {}) };
      next[key] = Number(input.value);
      FF.store.set({ config: next });
      paint();
    });
    paint();
    return h('div', { class: 'field' },
      h('div', { class: 'field-head' },
        h('label', { class: 'label', for: `ff-${key}`, text: label }),
        readout),
      input,
      h('p', { class: 'field-note', text: key === 'tolerance'
        ? 'Higher groups more faces together.' : null }));
  }

  /* ============================================================ the view */

  FF.views.scan = function (state) {
    const host = h('div', { class: 'view-body' });
    let disposed = false;
    let watching = false;
    let lastPreview = '';

    function configView() {
      const config = state.config || {};
      const dropped = state.droppedInputFolder;

      const children = [
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'Scan photos' }),
          h('p', { class: 'page-sub',
            text: 'Choose the folder to read and where the sorted folders should go. '
                + 'Nothing leaves this computer.' })),

        h('div', { class: 'scan-grid' },
          h('section', { class: 'panel' },
            h('div', { class: 'panel-head' },
              h('h2', { class: 'section-title', text: 'Folders' }),
              dropped
                ? h('span', { class: 'badge badge-accent', text: 'folder dropped' })
                : null),
            h('div', { class: 'panel-body scan-fields' },
              folderField('Photo folder', 'inputFolder', 'input',
                'The folder FaceFlow reads. It is never modified.'),
              folderField('Destination', 'outputFolder', 'output',
                'Sorted folders are created inside it.'),
              modeField())),

          h('section', { class: 'panel' },
            h('div', { class: 'panel-head' },
              h('h2', { class: 'section-title', text: 'Grouping' })),
            h('div', { class: 'panel-body scan-fields' },
              rangeField('Match strictness', 'tolerance', 0.2, 1, 0.05,
                (value) => Number(value).toFixed(2)),
              h('div', { class: 'field-row-2' },
                numberField('Minimum faces', 'minFaces', 1, 50, 1),
                numberField('Worker processes', 'workers', 0, 16, 1,
                  '0 chooses automatically')),
              callout({
                icon: 'sparkle',
                message: 'FaceFlow blends a whole-face fingerprint with one from the '
                      + 'eye region, which barely changes with age. That is what lets a '
                      + 'childhood photo and an adult one of the same person land in the '
                      + 'same group.',
              })))),

        h('div', { class: 'scan-launch' },
          h('button', {
            class: 'btn btn-primary btn-lg',
            type: 'button',
            onclick: start,
          }, icon('scan'), 'Start scan'),
          h('span', { class: 't-mute', text: 'Or drop a folder anywhere on this window.' }))];

      // A dropped folder fills the input immediately rather than making the
      // user find it in the form.
      if (dropped && dropped !== config.inputFolder) {
        const next = { ...config, inputFolder: dropped };
        FF.store.set({ config: next });
        const input = $('#ff-inputFolder');
        if (input) input.value = dropped;
      }

      return children;
    }

    function numberField(label, key, min, max, step, hint) {
      const input = h('input', {
        class: 'input', type: 'number', id: `ff-${key}`,
        min, max, step, value: (FF.store.state.config || {})[key] ?? 0,
      });
      input.addEventListener('input', () => {
        const next = { ...(FF.store.state.config || {}) };
        next[key] = Number(input.value);
        FF.store.set({ config: next });
      });
      return h('div', { class: 'field' },
        h('label', { class: 'label', for: `ff-${key}`, text: label }),
        input,
        hint ? h('p', { class: 'field-note', text: hint }) : null);
    }

    /* ------------------------------------------------- copy / move control */

    function modeField() {
      const current = (FF.store.state.config || {}).mode;
      return h('div', { class: 'field' },
        h('span', { class: 'label', id: 'ff-mode-label', text: 'When sorting' }),
        h('div', {
          class: 'segmented', role: 'radiogroup', 'aria-labelledby': 'ff-mode-label',
        }, ...['copy', 'move'].map((mode) => h('button', {
          type: 'button',
          role: 'radio',
          'aria-checked': String(current === mode),
          class: current === mode ? 'is-active' : '',
          onclick: () => {
            FF.store.set({ config: { ...(FF.store.state.config || {}), mode } });
            render();
          },
        }, icon(mode === 'copy' ? 'copy' : 'move', 'icon-sm'),
           mode === 'copy' ? 'Copy photos' : 'Move photos'))));
    }

    /* ------------------------------------------------------------ the scan */

    function scanningView(status) {
      const total = status.total || 0;
      const done = status.processed || 0;
      const pct = total ? Math.min(100, (done / total) * 100) : 0;
      const stats = status.stats || {};

      return [
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'Analysing your photos' }),
          h('p', { class: 'page-sub',
            text: 'Detecting faces and grouping them. This runs entirely on this computer.' })),

        h('div', { class: 'scan-progress' },
          h('div', { class: 'scan-count t-num' },
            fmt.count(done),
            h('span', { class: 'scan-count-total', text: ` / ${fmt.count(total)}` })),
          h('div', { class: 'progress progress-lg' },
            h('div', { class: 'progress-bar', style: { width: `${pct}%` } })),
          h('div', { class: 'scan-meta' },
            h('span', null, 'Currently analysing ',
              h('span', { class: 't-mono', text: status.current || '…' })),
            total
              ? h('span', { class: 't-num', text: `${pct.toFixed(0)}%` })
              : null)),

        h('div', { class: 'scan-live' },
          h('figure', { class: 'scan-preview card' },
            h('img', { id: 'scan-preview-img', alt: '', decoding: 'async' }),
            h('figcaption', { class: 't-mute', id: 'scan-preview-name', text: '—' })),
          h('ul', { class: 'scan-stats', id: 'scan-stats' },
            statRow('Faces detected', fmt.count(stats.faces || 0)),
            statRow('Groups discovered', fmt.count(status.clusters || 0)),
            statRow('Images with no face', fmt.count(stats.without_faces || 0)),
            statRow('Unreadable files', fmt.count(stats.unreadable || 0)),
            stats.img_per_s
              ? statRow('Speed', `${stats.img_per_s} img/s`)
              : null)),
      ];
    }

    const statRow = (label, value) => h('li', null,
      h('span', { class: 't-dim', text: label }),
      h('span', { class: 't-num', text: value }));

    /* -------------------------------------------------------- the filmstrip
       A row of what has just been read. A progress bar says how far; this
       says it is still working. */
    let strip = null;
    const pushStrip = (dataUrl) => {
      if (!strip || !dataUrl) return;
      const tile = h('div', { class: 'strip-tile' }, h('img', { src: dataUrl, alt: '', decoding: 'async' }));
      strip.prepend(tile);
      while (strip.children.length > 14) strip.lastElementChild.remove();
    };

    function render(status) {
      if (disposed) return;
      if (status && (status.scanning || status.organizing)) {
        const children = scanningView(status);
        if (status.organizing) {
          children.length = 0;
          children.push(organizingView(status));
        }
        if (!host.firstChild || host.dataset.phase !== (status.organizing ? 'org' : 'scan')) {
          host.dataset.phase = status.organizing ? 'org' : 'scan';
          strip = h('div', { class: 'scan-strip', 'aria-hidden': 'true' });
          children.push(h('section', { class: 'section' },
            h('h2', { class: 'section-title', text: 'Recently read' }), strip));
        } else {
          const existing = host.querySelector('.scan-strip');
          if (existing) strip = existing;
        }
        mount(host, children);
        return;
      }
      host.dataset.phase = '';
      mount(host, configView());
    }

    function organizingView(status) {
      const info = status.organize || {};
      const total = info.total || 0;
      const done = info.done || 0;
      const pct = total ? Math.min(100, (done / total) * 100) : 0;
      return [
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'Sorting your photos' }),
          h('p', { class: 'page-sub',
            text: 'Copying each photo into the folder for the person in it.' })),
        h('div', { class: 'scan-count t-num' },
          fmt.count(done),
          h('span', { class: 'scan-count-total', text: ` / ${fmt.count(total)}` })),
        h('div', { class: 'progress progress-lg' },
          h('div', { class: 'progress-bar', style: { width: `${pct}%` } })),
        h('p', { class: 't-mute', text: info.current ? `Placing ${info.current}` : 'Creating folders' }),
      ];
    }

    /* ------------------------------------------------------------ lifecycle */

    async function start() {
      const settings = FF.readSettings();
      if (!settings.input_folder) {
        const folder = await api.pickFolder('input');
        if (!folder) {
          toast('Choose a photo folder to scan', 'warning');
          return;
        }
        FF.store.set({ config: { ...FF.store.state.config, inputFolder: folder } });
        settings.input_folder = folder;
      }
      if (!settings.output_folder) {
        const folder = await api.pickFolder('output');
        if (!folder) {
          toast('Choose where the sorted folders should go', 'warning');
          return;
        }
        FF.store.set({ config: { ...FF.store.state.config, outputFolder: folder } });
        settings.output_folder = folder;
      }

      lastPreview = '';
      try {
        await api.scan(settings);
      } catch (error) {
        toast('Could not start the scan', 'error',
          { hint: error.hint, detail: error.detail });
        return;
      }
      FF.logActivity('Scan started', 'accent');
      if (!watching) {
        watching = true;
        FF.watch(onTick, 420);
      }
    }

    async function onTick(status) {
      if (disposed) return;
      render(status);

      // Live preview of the current photo, fetched through the bridge so the
      // binary never reaches the renderer's network stack.
      const name = status.current || '';
      if (name && name !== lastPreview && status.scanning) {
        lastPreview = name;
        const dataUrl = await api.preview(name);
        if (disposed) return;
        if (dataUrl) {
          const image = $('#scan-preview-img');
          const caption = $('#scan-preview-name');
          if (image) image.src = dataUrl;
          if (caption) caption.textContent = name;
          pushStrip(dataUrl);
        }
      }

      if (status.state === 'error') {
        mount(host,
          h('div', { class: 'page-head' },
            h('h1', { class: 'page-title', text: 'The scan did not finish' })),
          errorCallout(new FF.EngineError(status.error || 'Unknown error',
            'The engine reported a problem and stopped.'), {
            action: h('button', { class: 'btn', type: 'button', onclick: () => render() },
              'Change settings'),
          }));
        FF.logActivity('Scan failed', 'error');
        return;
      }

      if (!status.scanning && !status.organizing && !status.gallery
          && !status.results && status.clusters > 0 && !watching) {
        FF.logActivity(`Scan found ${status.clusters} groups`, 'success');
      }
    }

    render();
    return {
      node: host,
      teardown: () => {
        disposed = true;
        watching = false;
        FF.stopWatching();
      },
    };
  };
})(window.FF);