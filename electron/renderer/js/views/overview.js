/* ==========================================================================
   FaceFlow — Overview
   The first screen answers three questions: what does this do, where do my
   photos go, and what did it already find. Every figure below comes from
   /status; nothing is estimated or filled in.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon } = FF.dom;
  const { api, fmt } = FF;
  const { toast, emptyState, callout, errorCallout, statTile } = FF.ui;
  const { statTile: tile } = FF.components;

  FF.views = FF.views || {};

  FF.views.overview = function (state) {
    const host = h('div', { class: 'view-body' });
    let disposed = false;

    function render() {
      if (disposed) return;
      const config = state.config || {};
      const stats = state.stats || {};
      const results = state.lastResults;
      const hasScan = state.clusters > 0 || (stats.photos || 0) > 0;

      const children = [];

      /* ------------------------------------------------------------ hero */
      children.push(h('section', { class: 'hero' },
        h('h1', { class: 'page-title', text: 'Your photos, organized privately.' }),
        h('p', { class: 'page-sub',
          text: 'Find the people in your photos and organize them into folders. '
              + 'Everything happens on this computer.' }),
        h('div', { class: 'hero-actions' },
          h('button', {
            class: 'btn btn-primary btn-lg',
            type: 'button',
            onclick: () => FF.go('scan'),
          }, icon('scan'), 'Scan photos'),
          h('button', {
            class: 'btn btn-lg',
            type: 'button',
            onclick: () => FF.go('scan'),
          }, icon('folder-open'), 'Choose folder')),
        h('p', { class: 'hero-note' },
          icon('shield', 'icon-sm'),
          ' No photo, face or name is uploaded. The app makes no network requests '
          + 'and needs no account.')));

      /* ----------------------------------------------------------- stats */
      children.push(h('section', { class: 'section' },
        h('div', { class: 'stats' },
          tile('Photos scanned', hasScan ? fmt.count(stats.photos || 0) : '—'),
          tile('Faces detected', hasScan ? fmt.count(stats.faces || 0) : '—'),
          tile('People identified', hasScan ? fmt.count(state.named || 0) : '—'),
          tile('Groups discovered', hasScan ? fmt.count(state.clusters || 0) : '—',
            hasScan && state.unknown
              ? `${fmt.count(state.unknown)} still unnamed` : null))));

      /* --------------------------------------------------- input / output */
      children.push(h('section', { class: 'section' },
        h('div', { class: 'section-head' },
          h('div', null,
            h('h2', { class: 'section-title', text: 'Folders' }),
            h('p', { class: 'section-note',
              text: 'Where FaceFlow reads from and where it writes to.' })),
          h('div', { class: 'section-actions' },
            h('button', { class: 'btn btn-sm', type: 'button', onclick: () => FF.go('settings') },
              icon('settings', 'icon-sm'), 'Settings'))),
        h('div', { class: 'panel' },
          h('div', { class: 'panel-body' },
            h('dl', { class: 'rows' },
              h('div', null,
                h('dt', { text: 'Photo folder' }),
                h('dd', null, h('span', { class: 'input-mono truncate',
                  text: config.inputFolder || 'Not chosen' }))),
              h('div', null,
                h('dt', { text: 'Destination' }),
                h('dd', null, h('span', { class: 'input-mono truncate',
                  text: config.outputFolder || 'Not chosen' }))),
              h('div', null,
                h('dt', { text: 'When sorting' }),
                h('dd', { text: config.mode === 'move' ? 'Move originals' : 'Copy (originals untouched)' })))))));

      /* --------------------------------------------------- last run result */
      if (results) {
        const folderRows = Object.entries(results.folders || {})
          .sort((a, b) => b[1] - a[1]);
        children.push(h('section', { class: 'section' },
          h('div', { class: 'section-head' },
            h('div', null,
              h('h2', { class: 'section-title', text: 'Last organised' }),
              h('p', { class: 'section-note',
                text: `${fmt.plural(results.files_placed, 'photo')} in `
                    + `${fmt.plural(Object.keys(results.folders || {}).length, 'folder')}.` })),
            h('div', { class: 'section-actions' },
              h('button', {
                class: 'btn btn-sm', type: 'button',
                onclick: () => api.openPath(results.output_folder),
              }, icon('folder-open', 'icon-sm'), 'Open folder'),
              h('button', { class: 'btn btn-sm', type: 'button', onclick: () => FF.go('photos') },
                icon('image', 'icon-sm'), 'Review groups'))),
          folderRows.length
            ? h('div', { class: 'folder-chips stagger' },
                folderRows.map(([name, count], index) => h('div', {
                  class: 'folder-chip', style: { '--i': String(index) },
                },
                  icon('folder', 'icon-sm'),
                  h('span', { class: 'truncate', text: name }),
                  h('span', { class: 't-num t-dim', text: fmt.count(count) }))))
            : h('p', { class: 't-dim', text: 'No folders were created in that run.' })));
      }

      /* --------------------------------------------------- next best thing */
      if (!hasScan) {
        children.push(h('section', { class: 'section' },
          emptyState({
            icon: 'scan',
            title: 'No scan has been completed',
            note: 'Point FaceFlow at a folder of photos and it will find the people '
                + 'in them, then let you name each group.',
            action: h('button', {
              class: 'btn btn-primary', type: 'button', onclick: () => FF.go('scan'),
            }, icon('scan'), 'Scan photos'),
          })));
      } else if (state.unknown > 0) {
        children.push(h('section', { class: 'section' },
          callout({
            kind: 'accent',
            icon: 'tag',
            title: `${fmt.plural(state.unknown, 'group')} still unnamed`,
            message: 'Naming a group decides which folder its photos land in. '
                   + 'Groups you leave unnamed go to '
                   + `${(state.config || {}).unknownFolder || '_unknown'}/.`,
            action: h('button', {
              class: 'btn btn-sm', type: 'button', onclick: () => FF.go('photos'),
            }, 'Name groups'),
          })));
      }

      mount(host, children);
    }

    render();
    // A fresh status keeps the overview honest without a full reload.
    const off = FF.store.subscribe(() => { if (!disposed) render(); });
    api.status().then((status) => { if (!disposed) FF.applyStatusToShell(status); }).catch(() => {});

    return { node: host, teardown: () => { disposed = true; off(); } };
  };
})(window.FF);