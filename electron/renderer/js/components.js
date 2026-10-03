/* ==========================================================================
   FaceFlow — shared components
   The cluster card is the centrepiece of the review screen, so it gets the
   most care: a real face montage, lazy thumbnails, and actions that state
   their outcome rather than opening a mystery.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, icon, debounce, stagger } = FF.dom;
  const { fmt } = FF;
  const { toast } = FF.ui;

  /* ============================================================ face tile
     The engine already returns each face as an inline data-URL thumbnail, so
     no network request is involved. `loading=lazy` still matters: a 1,000-face
     cluster should not decode a thousand base64 payloads on first paint. */

  function faceTile(face, options) {
    const settings = options || {};
    const image = h('img', {
      src: face.thumb || '',
      alt: settings.decorative ? '' : `Face from ${face.photo || 'a photo'}`,
      loading: settings.lazy === false ? 'eager' : 'lazy',
      decoding: 'async',
    });
    if (!face.thumb) image.classList.add('is-missing');
    const tile = h('button', {
      class: 'face-tile',
      type: 'button',
      title: face.photo ? `${face.photo} — click to enlarge` : 'Click to enlarge',
      onclick: (event) => { event.stopPropagation(); if (settings.onOpen) settings.onOpen(face); },
    }, image);
    return tile;
  }

  /* ========================================================= cluster card
     One person-sized group. Front face shows the montage and the actions;
     the back face lists every photo and the destination, for when someone
     needs to check what is in there. */

  function clusterCard(cluster, handlers) {
    const acts = handlers || {};
    const faces = cluster.faces || [];
    const named = Boolean(cluster.name);
    const shown = faces.slice(0, 6);
    const extra = Math.max(0, faces.length - shown.length);

    const nameNode = h('div', { class: 'cluster-name truncate', text: named ? cluster.name : 'Unnamed group' });
    const metaNode = h('div', { class: 'cluster-meta' },
      `${fmt.plural(cluster.size, 'face')} · ${fmt.plural(cluster.photos, 'photo')}`);

    const back = h('div', { class: 'cluster-face cluster-back' },
      h('div', { class: 'cluster-name truncate', text: named ? cluster.name : `Group ${cluster.id + 1}` }),
      h('dl', { class: 'rows cluster-rows' },
        row('Destination', named ? `${cluster.name}/` : (acts.unknownFolder || '_unknown') + '/'),
        row('Photos', fmt.count(cluster.photos)),
        row('Faces', fmt.count(cluster.size)),
        row('Status', named
          ? (cluster.auto ? 'remembered from a previous run' : 'named just now')
          : 'goes to the unknown folder')),
      faces.length
        ? h('ul', { class: 'cluster-files' },
            faces.map((face) => h('li', { class: 'truncate', title: face.photo, text: face.photo })))
        : h('p', { class: 't-mute', text: 'No file list available.' }),
      h('button', {
        class: 'btn btn-ghost btn-sm',
        type: 'button',
        onclick: (event) => { event.stopPropagation(); if (acts.onFlip) acts.onFlip(card, false); },
      }, icon('arrow-left', 'icon-sm'), 'Back'));

    const front = h('div', { class: 'cluster-face cluster-front' },
      h('div', { class: 'cluster-montage', dataset: { count: String(Math.min(shown.length, 6)) } },
        shown.map((face) => faceTile(face, { onOpen: acts.onOpenPhoto })),
        extra
          ? h('div', { class: 'face-more', text: `+${fmt.count(extra)}` })
          : null),
      h('div', { class: 'cluster-id-row' },
        nameNode,
        named
          ? h('span', { class: `badge ${cluster.auto ? 'badge-accent' : 'badge-success'}`,
                        text: cluster.auto ? 'remembered' : 'named' })
          : h('span', { class: 'badge', text: 'unnamed' })),
      metaNode,
      // When the engine could not fingerprint the eye region, this group was
      // matched on whole faces alone — which is exactly when a manual link is
      // the right tool, so the card says so rather than leaving it to be
      // discovered.
      typeof cluster.eye_coverage === 'number' && cluster.eye_coverage < 0.999
        ? h('div', { class: 'cluster-note',
                     text: cluster.eye_coverage > 0
                       ? `${Math.round(cluster.eye_coverage * 100)}% eye-region detail`
                       : 'Low eye-region detail — may be a childhood group' })
        : null,
      h('div', { class: 'cluster-actions' },
        h('button', {
          class: 'btn btn-sm', type: 'button',
          onclick: (event) => { event.stopPropagation(); if (acts.onName) acts.onName(cluster); },
        }, icon('tag', 'icon-sm'), named ? 'Rename' : 'Name'),
        h('button', {
          class: 'btn btn-sm', type: 'button',
          title: 'Merge this group with another (they are the same person)',
          onclick: (event) => { event.stopPropagation(); if (acts.onMerge) acts.onMerge(cluster); },
        }, icon('merge', 'icon-sm'), 'Merge'),
        h('button', {
          class: 'btn btn-sm', type: 'button',
          title: acts.splitReason || 'Not available yet',
          disabled: acts.onSplit ? false : true,
          onclick: (event) => { event.stopPropagation(); if (acts.onSplit) acts.onSplit(cluster); },
        }, icon('split', 'icon-sm'), 'Split'),
        h('button', {
          class: 'btn btn-ghost btn-sm btn-icon', type: 'button',
          'aria-label': `Show details for ${named ? cluster.name : `group ${cluster.id + 1}`}`,
          title: 'Details',
          onclick: (event) => { event.stopPropagation(); if (acts.onFlip) acts.onFlip(card, true); },
        }, icon('eye', 'icon-sm'))));

    const card = h('article', {
      class: `cluster ${named ? 'is-named' : ''} ${acts.picked ? 'is-picked' : ''}`,
      dataset: { clusterId: String(cluster.id) },
      tabindex: '0',
      role: 'group',
      'aria-label': named
        ? `${cluster.name}, ${fmt.plural(cluster.photos, 'photo')}`
        : `Unnamed group ${cluster.id + 1}, ${fmt.plural(cluster.photos, 'photo')}`,
    },
      h('div', { class: 'cluster-inner' }, front, back));

    // Clicking anywhere that is not a control is the primary action for this
    // card: pick it (merge in progress) or name it. Keyboard users get the
    // same route via Enter/Space.
    const primary = () => {
      if (acts.onSelect) acts.onSelect();
      else if (acts.onName) acts.onName(cluster);
    };
    card.addEventListener('click', (event) => {
      if (event.target.closest('button, a, input')) return;
      primary();
    });
    card.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        if (event.target.closest('button, a, input')) return;
        event.preventDefault();
        primary();
      }
    });

    return card;
  }

  const row = (label, value) => h('div', null,
    h('dt', { text: label }), h('dd', { class: 'truncate', text: value }));

  /* ============================================================ photo grid
     Shared by Person detail, Relationships results and the cluster detail
     view. Photos arrive as filenames; the engine serves them as data URLs via
     the bridge, so each one is fetched lazily and only when scrolled near. */

  function photoGrid(photos, options) {
    const settings = options || {};
    const loaded = new Map();
    const nodes = new Map();

    // Reuse one Image per photo so the lightbox can page instantly.
    const imageFor = (name) => {
      if (loaded.has(name)) return loaded.get(name);
      const image = new Image();
      image.decoding = 'async';
      image.alt = name;
      image.loaded = false;
      FF.api.photo(name).then((url) => {
        if (!url) return;
        image.src = url;
        image.loaded = true;
        const node = nodes.get(name);
        if (node) node.style.setProperty('--has-image', '1');
      }).catch(() => { /* a missing photo leaves its placeholder */ });
      loaded.set(name, image);
      return image;
    };

    const tiles = photos.map((name, index) => {
      const tile = h('button', {
        class: 'photo-tile',
        type: 'button',
        title: name,
        'aria-label': `Open ${name}`,
        onclick: () => settings.onOpen
          ? settings.onOpen(name, index)
          : FF.ui.openLightbox({ photos: photos.map(imageFor), photo: name }),
      }, h('span', { class: 'photo-thumb' }));
      nodes.set(name, tile);
      // Only warm the first row immediately; the rest load as they approach
      // the viewport, which is what keeps a 4,000-photo grid responsive.
      if (index < settings.eager || 8) imageFor(name);
      return tile;
    });

    if ('IntersectionObserver' in window && tiles.length) {
      const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const tile = entry.target;
          observer.unobserve(tile);
          const name = tile.getAttribute('data-photo');
          if (name) imageFor(name);
        }
      }, { rootMargin: '320px' });
      photos.forEach((name, index) => {
        const tile = tiles[index];
        tile.dataset.photo = name;
        if (index >= (settings.eager || 8)) observer.observe(tile);
      });
    } else {
      photos.forEach((name) => imageFor(name));
    }

    if (settings.cap && tiles.length > settings.cap) {
      const shown = tiles.slice(0, settings.cap);
      const more = h('p', { class: 't-dim photo-more',
        text: `Showing the first ${fmt.count(settings.cap)} of ${fmt.count(tiles.length)}.` });
      return h('div', null, h('div', { class: 'photo-grid' }, shown), more);
    }

    const grid = h('div', { class: 'photo-grid stagger' }, tiles);
    stagger(grid, tiles, 24);
    return grid;
  }

  /* ============================================================== person row
     Used in the People picker and in search results. */

  function personRow(person, options) {
    const settings = options || {};
    return h('label', { class: `person-row ${settings.selected ? 'is-selected' : ''}` },
      h('input', {
        type: settings.checkbox === false ? 'radio' : 'checkbox',
        name: settings.name || 'people',
        checked: Boolean(settings.checked),
        onchange: settings.onChange,
      }),
      h('span', { class: 'person-row-name truncate', text: person.name }),
      settings.photos !== undefined
        ? h('span', { class: 'person-row-count t-num', text: fmt.count(person.photos) })
        : null);
  }

  /* ============================================================== stat tile
     A single figure with its label. Used on the overview only, where four
     numbers in a row is easier to read than a ledger. */

  function statTile(label, value, options) {
    const settings = options || {};
    return h('div', { class: 'stat' },
      h('div', { class: 'stat-value t-num', text: value }),
      h('div', { class: 'stat-label', text: label }),
      settings.note ? h('div', { class: 'stat-note', text: settings.note }) : null);
  }

  FF.components = {
    faceTile, clusterCard, photoGrid, personRow, statTile,
  };
})(window.FF);