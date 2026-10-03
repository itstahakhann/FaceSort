/* ==========================================================================
   FaceFlow — Photos (cluster review)
   The screen where the user corrects the engine's grouping. Everything here
   is about trust: show what the engine decided, make a correction cheap, and
   never make the user reload to see the result of their own edit.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon, debounce, stagger } = FF.dom;
  const { api, fmt } = FF;
  const { toast, dialog, confirm, emptyState, callout, errorCallout } = FF.ui;
  const { clusterCard } = FF.components;

  FF.views = FF.views || {};

  /** Split is not implemented by the engine. Rather than shipping a button
   *  that silently does nothing, the control is present, disabled, and says
   *  why. When POST /split_cluster lands this becomes a live action. */
  const SPLIT_REASON =
    'Splitting a group is not supported by the local engine yet. Use Merge to '
    + 'join two groups the engine split apart, and Skip to send a group to the '
    + 'unknown folder.';

  FF.views.photos = function (state) {
    const host = h('div', { class: 'view-body' });
    let disposed = false;

    const local = {
      clusters: [],
      names: new Map(),
      query: '',
      sort: 'size',
      // Merge is a two-step selection: pick the first group, then the second.
      merging: false,
      picks: [],
      loading: true,
      error: null,
    };

    /* ------------------------------------------------------------- loading */

    async function load() {
      local.loading = true;
      local.error = null;
      render();
      try {
        const payload = await api.clusters();
        local.clusters = payload.clusters || [];
        local.names = new Map(local.clusters
          .filter((cluster) => cluster.name)
          .map((cluster) => [cluster.id, cluster.name]));
        local.loading = false;
        render();
      } catch (error) {
        local.loading = false;
        local.error = error;
        render();
      }
    }

    const visible = () => {
      const query = local.query.trim().toLowerCase();
      let list = local.clusters.filter((cluster) => {
        if (!query) return true;
        const name = (cluster.name || '').toLowerCase();
        return name.includes(query) || `group ${cluster.id + 1}`.includes(query);
      });
      if (local.sort === 'size') {
        list = list.slice().sort((a, b) => b.size - a.size);
      } else if (local.sort === 'name') {
        list = list.slice().sort((a, b) =>
          (a.name || `~${a.id}`).localeCompare(b.name || `~${b.id}`));
      } else if (local.sort === 'photos') {
        list = list.slice().sort((a, b) => b.photos - a.photos);
      }
      return list;
    };

    const unnamedCount = () => local.clusters.filter((cluster) => !cluster.name).length;

    /* -------------------------------------------------------------- naming */

    async function nameDialog(cluster) {
      const existing = cluster.name || '';
      const input = h('input', {
        class: 'input input-name',
        type: 'text',
        id: 'ff-name-input',
        placeholder: 'e.g. Alex',
        value: existing,
        autocomplete: 'off',
        maxlength: '60',
        'data-autofocus': 'true',
      });
      const error = h('p', { class: 'field-error', hidden: true });

      const submit = async (close) => {
        const value = input.value.trim();
        if (!value) {
          error.textContent = 'Enter a name, or use Skip to send this group to the '
                           + `${(FF.store.state.config || {}).unknownFolder || '_unknown'} folder.`;
          error.hidden = false;
          input.focus();
          return;
        }
        if (existing) {
          try {
            await api.nameCluster(cluster.id, value);
          } catch (caught) {
            error.textContent = (caught && caught.hint) || 'The name could not be saved.';
            error.hidden = false;
            return;
          }
        } else {
          const ok = await mergeFirst(2);
          if (!ok) { close(); return; }
        }
        // Update in place. A reload here would lose the user's scroll position
        // and re-fetch every face crop for a one-word change.
        cluster.name = value;
        local.names.set(cluster.id, value);
        render();
        close();
        FF.announce(`${value} named`);
        toast(existing ? 'Cluster renamed successfully' : `${value} named`, 'success');
        FF.logActivity(`${value} named`, 'success');
      };

      dialog({
        title: existing ? 'Rename this person' : 'Who is this?',
        subtitle: `${fmt.plural(cluster.size, 'face')} across `
                + `${fmt.plural(cluster.photos, 'photo')}`,
        body: [
          h('div', { class: 'name-preview' },
            (cluster.faces || []).slice(0, 4).map((face) => h('img', {
              src: face.thumb || '', alt: '', decoding: 'async',
            }))),
          h('div', { class: 'field' },
            h('label', { class: 'label', for: 'ff-name-input', text: 'Person name' }),
            input,
            error),
        ],
        footer: (close) => [
          h('button', {
            class: 'btn', type: 'button',
            onclick: async () => {
              close();
              await api.nameCluster(cluster.id, '').catch(() => {});
              cluster.name = '';
              local.names.delete(cluster.id);
              render();
              FF.logActivity('Group skipped', '');
            },
          }, `Send to ${(FF.store.state.config || {}).unknownFolder || '_unknown'}`),
          h('span', { class: 'spacer' }),
          h('button', { class: 'btn', type: 'button', onclick: () => close() }, 'Cancel'),
          h('button', {
            class: 'btn btn-primary', type: 'button',
            onclick: () => submit(close),
          }, 'Save'),
        ],
      });

      // Enter saves without reaching for the mouse.
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') { event.preventDefault(); submit(() => {}); }
      });
    }

    /* --------------------------------------------------------------- merge
       Merging is the corrective action this product exists for: the engine
       split one person into two groups, so the user asserts the truth. */

    async function mergeFirst(limit) {
      const first = local.picks[0];
      if (first === undefined) return false;
      const label = (id) => {
        const cluster = local.clusters.find((item) => item.id === id);
        return cluster ? (cluster.name || `Group ${cluster.id + 1}`) : `#${id}`;
      };
      const target = limit === 2
        ? { title: 'Name this person', subtitle: 'Their photos will be combined into one group.' }
        : { title: 'Who is it?', subtitle: 'Their photos will be combined into one group.' };

      const input = h('input', {
        class: 'input input-name', type: 'text', id: 'ff-merge-name',
        placeholder: 'e.g. Alex', autocomplete: 'off',
      });
      const error = h('p', { class: 'field-error', hidden: true });

      dialog({
        ...target,
        body: [
          h('div', { class: 'merge-summary' },
            chip(label(first)),
            icon('link'),
            chip(label(local.picks[1]))),
          h('div', { class: 'field' },
            h('label', { class: 'label', for: 'ff-merge-name', text: 'Person name' }),
            input,
            h('p', { class: 'field-note',
              text: 'Optional. A name is remembered, so future scans group them on their own.' }),
            error),
        ],
        footer: (close) => [
          h('span', { class: 'spacer' }),
          h('button', { class: 'btn', type: 'button', onclick: () => close() }, 'Cancel'),
          h('button', {
            class: 'btn btn-primary', type: 'button',
            onclick: async () => {
              try {
                await api.mergeClusters(first, local.picks[1], input.value.trim());
              } catch (caught) {
                error.textContent = (caught && caught.hint)
                  || 'Those two groups could not be merged.';
                error.hidden = false;
                return;
              }
              const name = input.value.trim();
              close();
              local.picks = [];
              local.merging = false;
              FF.logActivity('Groups merged', 'success');
              toast(name ? `${name} merged` : 'Groups merged', 'success');
              await load();
            },
          }, 'Merge'),
        ],
      });
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
      });
    }

    const chip = (text) => h('span', { class: 'merge-chip', text });

    function togglePick(cluster) {
      const index = local.picks.indexOf(cluster.id);
      if (index > -1) {
        local.picks.splice(index, 1);
      } else if (local.picks.length >= 2) {
        // Replace the older selection: merging is always about two groups.
        local.picks.shift();
        local.picks.push(cluster.id);
      } else {
        local.picks.push(cluster.id);
      }
      render();
      if (local.picks.length === 2) mergeFirst(2);
    }

    function askMergeStart(cluster) {
      local.merging = true;
      local.picks = [cluster.id];
      render();
      toast('Now pick the second group to merge with it', 'info', { ttl: 4200 });
    }

    function cancelMerge() {
      local.merging = false;
      local.picks = [];
      render();
    }

    /* -------------------------------------------------------------- render */

    const onQuery = debounce((value) => { local.query = value; render(); }, 140);

    function render() {
      if (disposed) return;
      const list = visible();
      const children = [];

      children.push(h('div', { class: 'page-head' },
        h('h1', { class: 'page-title', text: 'People discovered' }),
        h('p', { class: 'page-sub' },
          `${fmt.plural(local.clusters.length, 'group')} · `
          + `${fmt.plural(unnamedCount(), 'unnamed', 'unnamed')}. `
          + 'Give each group a name, or merge two groups that are really the same person.')));

      if (local.merging) {
        children.push(callout({
          kind: 'accent',
          icon: 'merge',
          title: 'Merging two groups',
          message: local.picks.length === 1
            ? 'Now click the other group that is the same person. Click this message to cancel.'
            : 'Pick the second group.',
          action: h('button', { class: 'btn btn-sm', type: 'button', onclick: cancelMerge },
            'Cancel merge'),
        }));
      }

      if (local.error) {
        children.push(errorCallout(local.error, {
          action: h('button', { class: 'btn btn-sm', type: 'button', onclick: load },
            'Try again'),
        }));
      }

      /* ------------------------------------------------------- toolbar */
      children.push(h('div', { class: 'review-toolbar' },
        h('div', { class: 'search review-search' },
          icon('search'),
          h('input', {
            class: 'input', type: 'search', id: 'ff-cluster-search',
            placeholder: 'Search by name', value: local.query,
            'aria-label': 'Search groups by name',
            oninput: (event) => onQuery(event.target.value),
          })),
        h('div', { class: 'segmented', role: 'group', 'aria-label': 'Sort groups' },
          ...[['size', 'Largest'], ['name', 'Name'], ['photos', 'Photos']].map(([value, label]) =>
            h('button', {
              type: 'button',
              class: local.sort === value ? 'is-active' : '',
              'aria-pressed': String(local.sort === value),
              onclick: () => { local.sort = value; render(); },
            }, label))),
        h('span', { class: 'spacer' }),
        h('button', {
          class: 'btn btn-primary', type: 'button',
          disabled: unnamedCount() > 0 || !local.clusters.length,
          title: unnamedCount() > 0
            ? `Name or skip the remaining ${unnamedCount()} group(s) first`
            : 'Copy photos into the folders',
          onclick: organise,
        }, icon('sort'), 'Sort into folders')));

      /* ---------------------------------------------------------- body */
      if (local.loading) {
        children.push(FF.ui.skeletonGrid(6, 'cluster-grid'));
      } else if (!local.clusters.length) {
        children.push(emptyState({
          icon: 'scan',
          title: 'No groups yet',
          note: 'Run a scan and the people it finds will appear here.',
          action: h('button', {
            class: 'btn btn-primary', type: 'button', onclick: () => FF.go('scan'),
          }, icon('scan'), 'Scan photos'),
        }));
      } else if (!list.length) {
        children.push(emptyState({
          icon: 'search',
          title: `Nothing matches “${local.query}”`,
          note: 'Try part of a name, or a group number.',
          action: h('button', {
            class: 'btn', type: 'button',
            onclick: () => { local.query = ''; render(); },
          }, 'Clear search'),
        }));
      } else {
        const grid = h('div', { class: 'cluster-grid stagger' },
          list.map((cluster) => clusterCard(cluster, {
            picked: local.picks.includes(cluster.id),
            unknownFolder: (FF.store.state.config || {}).unknownFolder,
            splitReason: SPLIT_REASON,
            // While a merge is in progress a click means "this is the other
            // group", not "open the name box". Without this the second pick
            // was unreachable.
            onSelect: local.merging ? () => togglePick(cluster) : null,
            onName: local.merging ? null : nameDialog,
            onMerge: askMergeStart,
            onSplit: null,          // engine has no split endpoint
            onFlip: flipCard,
            onOpenPhoto: (face) => openFace(cluster, face),
          })));
        stagger(grid, [...grid.children], 14);
        children.push(grid);
      }

      mount(host, children);
    }

    function flipCard(card, forward) {
      card.classList.toggle('is-flipped', forward);
    }

    /* --------------------------------------------------------- lightbox
       The engine scopes /photo by filename, and the cluster's faces already
       carry the filenames, so paging within one group needs no extra request. */
    function openFace(cluster, face) {
      const faces = (cluster.faces || []).filter((item) => item.thumb);
      if (!faces.length) return;
      const start = faces.indexOf(face);
      const photos = faces.map((item) => ({
        name: item.photo,
        src: item.thumb,
        load: () => api.photo(item.photo).then((url) => {
          if (!url) return;
          const image = new Image();
          image.src = url;
        }),
      }));
      FF.ui.openLightbox({ photos, photo: photos[Math.max(0, start)] });
    }

    /* ------------------------------------------------------------ organize */

    async function organise() {
      const config = FF.store.state.config || {};
      const unknown = config.unknownFolder || '_unknown';
      const skipped = unnamedCount();
      const totalPhotos = local.clusters.reduce((sum, cluster) => sum + cluster.photos, 0);

      const proceed = await confirm({
        title: 'Create the folders now?',
        subtitle: `${fmt.plural(local.clusters.length, 'folder')} will be created in `
                + `${config.outputFolder || 'the destination folder'}.`,
        confirmLabel: 'Create folders',
        body: [
          h('dl', { class: 'rows' },
            row('Named people', fmt.count(local.clusters.length - skipped)),
            skipped ? row(`Unnamed groups, into ${unknown}/`, fmt.count(skipped)) : null,
            row(config.mode === 'move' ? 'Photos moved' : 'Photos copied',
                fmt.count(totalPhotos))),
          config.mode === 'move'
            ? h('p', { class: 'field-note', text: 'Originals will be moved, not copied.' })
            : null,
        ].filter(Boolean),
      });
      if (!proceed) return;

      try {
        await api.organize(config.mode);
      } catch (error) {
        toast('Could not start sorting', 'error', { hint: error.hint, detail: error.detail });
        return;
      }
      FF.logActivity('Sorting started', 'accent');
      FF.stopWatching();
      FF.watch((status) => {
        if (disposed) return;
        if (status.state === 'error') {
          toast('Sorting failed', 'error', { hint: status.error });
          return;
        }
        if (status.results) {
          const results = status.results;
          FF.store.set({ lastResults: results });
          FF.logActivity(
            `${results.files_placed} photos in ${Object.keys(results.folders || {}).length} folders`,
            'success');
          showDone(results);
        }
      }, 420);
    }

    function showDone(results) {
      const folderRows = Object.entries(results.folders || {}).sort((a, b) => b[1] - a[1]);
      mount(host,
        h('div', { class: 'done-mark' }, icon('check', 'icon-lg')),
        h('div', { class: 'page-head center' },
          h('h1', { class: 'page-title', text: 'All sorted' }),
          h('p', { class: 'page-sub',
            text: `${fmt.plural(results.files_placed, 'photo')} copied into `
                + `${fmt.plural(folderRows.length, 'folder')}.` })),
        h('dl', { class: 'rows rows-wide' },
          row('Photos placed', fmt.count(results.files_placed)),
          row('Folders created', fmt.count(folderRows.length)),
          row('Unknown groups', fmt.count(results.unknown_placed || 0)),
          results.sources_removed ? row('Originals moved', fmt.count(results.sources_removed)) : null,
          (results.errors && results.errors.length)
            ? row('Could not be placed', fmt.count(results.errors.length)) : null),
        h('div', { class: 'scan-launch' },
          h('button', {
            class: 'btn btn-primary', type: 'button',
            onclick: () => api.openPath(results.output_folder),
          }, icon('folder-open'), 'Open output folder'),
          h('button', {
            class: 'btn', type: 'button',
            onclick: () => { local.clusters = []; FF.go('overview'); },
          }, 'Back to overview'),
          h('button', {
            class: 'btn', type: 'button', onclick: () => FF.go('gallery'),
          }, icon('camera'), 'Build a gallery')));
    }

    const row = (label, value) => h('div', null,
      h('dt', { text: label }), h('dd', { text: value }));

    load();
    return { node: host, teardown: () => { disposed = true; onQuery.cancel(); } };
  };
})(window.FF);