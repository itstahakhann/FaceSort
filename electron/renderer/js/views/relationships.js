/* ==========================================================================
   FaceFlow — Relationships ("People together")
   Answers the question the co-occurrence table is really asking: who was in
   this photo with whom? The matrix is presented as a set of pairs rather
   than as a database grid, because that is what a person can act on.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon, debounce, stagger } = FF.dom;
  const { api, fmt } = FF;
  const { toast, emptyState, callout, errorCallout } = FF.ui;
  const { photoGrid, personRow } = FF.components;

  FF.views = FF.views || {};

  FF.views.relationships = function (state) {
    const host = h('div', { class: 'view-body' });
    const local = {
      people: [],
      query: '',
      picks: [],
      result: null,
      unknown: [],
      loading: true,
      searching: false,
      error: null,
      coOccurrence: null,
      coOccurrenceError: null,
    };
    let disposed = false;

    // A person picked on the People page arrives here pre-selected.
    if (Array.isArray(FF.pendingRelationshipPick) && FF.pendingRelationshipPick.length) {
      local.picks = FF.pendingRelationshipPick.slice();
      FF.pendingRelationshipPick = null;
    }

    async function load() {
      local.loading = true;
      render();
      try {
        const payload = await api.peopleList();
        local.people = payload.people || [];
        local.loading = false;
        render();
        if (local.picks.length) search();
        loadCoOccurrence();
      } catch (error) {
        local.loading = false;
        local.error = error;
        render();
      }
    }

    async function loadCoOccurrence() {
      try {
        local.coOccurrence = await api.coOccurrence();
      } catch (error) {
        // A missing matrix should not take the whole view down; the pairwise
        // search above it still works.
        local.coOccurrenceError = error;
      }
      if (!disposed) render();
    }

    /* ------------------------------------------------------------- search */

    async function search() {
      if (!local.picks.length) {
        local.result = null;
        render();
        return;
      }
      local.searching = true;
      render();
      try {
        const payload = await api.intersection(local.picks);
        local.result = payload;
        local.unknown = payload.unknown || [];
      } catch (error) {
        local.error = error;
      }
      local.searching = false;
      if (!disposed) render();
    }

    function toggle(name) {
      const index = local.picks.indexOf(name);
      if (index > -1) local.picks.splice(index, 1);
      else local.picks.push(name);
      search();
    }

    const onQuery = debounce((value) => { local.query = value; render(); }, 140);

    const filteredPeople = () => {
      const query = local.query.trim().toLowerCase();
      return local.people.filter((person) => person.name.toLowerCase().includes(query));
    };

    /* -------------------------------------------------------- co-occurrence
       Rendered as a ranked list of pairs. A matrix grid is compact but reads
       as a spreadsheet; "these two appear in 14 photos together" is the
       insight, and it is one click from a photo grid. */
    function pairsView() {
      const payload = local.coOccurrence;
      if (!payload) return null;
      const names = payload.names || [];
      const pairs = (payload.pairs || [])
        .filter((pair) => pair.photos > 0)
        .sort((a, b) => b.photos - a.photos);

      if (!pairs.length) {
        return h('p', { class: 't-dim', text: 'No shared photos recorded yet.' });
      }

      const max = payload.max || pairs[0].photos || 1;
      return h('ol', { class: 'pair-list stagger' },
        pairs.slice(0, 24).map((pair, index) => h('li', {
          class: 'pair',
          // The bar width carries the magnitude, so it needs a unit: a bare
          // number is an invalid `width` and silently falls back to auto.
          // Minimum 8% so a rare pair is still visibly a bar rather than a sliver.
          style: {
            '--i': String(index),
            '--weight': `${Math.max(8, (pair.photos / max) * 100).toFixed(1)}%`,
          },
        },
          h('button', {
            class: 'pair-btn', type: 'button',
            onclick: () => { local.picks = [pair.a, pair.b]; search(); },
          },
            h('span', { class: 'pair-names' },
              h('span', { class: 'pair-name', text: pair.a }),
              icon('link', 'icon-sm'),
              h('span', { class: 'pair-name', text: pair.b })),
            h('span', { class: 'pair-bar', 'aria-hidden': 'true' },
              h('span', { class: 'pair-bar-fill' })),
            h('span', { class: 'pair-count t-num',
              text: `${fmt.plural(pair.photos, 'photo')}` })))));
    }

    /* -------------------------------------------------------------- render */

    function render() {
      if (disposed) return;
      const result = local.result;
      const photos = (result && result.photos) || [];
      const children = [
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'People together' }),
          h('p', { class: 'page-sub',
            text: 'Pick two or more people to see only the photos where every one of '
                + 'them appears.' })),
      ];

      if (local.error) children.push(errorCallout(local.error));

      if (local.loading) {
        children.push(FF.ui.skeletonGrid(4, 'people-grid'));
        mount(host, children);
        return;
      }

      if (!local.people.length) {
        children.push(emptyState({
          icon: 'link',
          title: 'No relationships found',
          note: local.error
            ? 'The name database is unavailable right now.'
            : 'Nobody has been named yet. Relationships are drawn from the names you '
              + 'give your groups, so name a few first.',
          action: h('button', {
            class: 'btn btn-primary', type: 'button', onclick: () => FF.go('photos'),
          }, icon('image'), 'Review groups'),
        }));
        mount(host, children);
        return;
      }

      /* --------------------------------------------------------- the picker */
      children.push(h('section', { class: 'panel' },
        h('div', { class: 'panel-head' },
          h('h2', { class: 'section-title', text: 'Select people' }),
          h('span', { class: 't-mute', id: 'ff-pick-count',
            text: local.picks.length ? `${fmt.plural(local.picks.length, 'person', 'people')} selected` : 'None selected' })),
        h('div', { class: 'panel-body' },
          h('div', { class: 'search', style: { marginBottom: 'var(--s4)' } },
            icon('search'),
            h('input', {
              class: 'input', type: 'search', id: 'ff-rel-search',
              placeholder: 'Search people', value: local.query,
              'aria-label': 'Search people to select',
              oninput: (event) => onQuery(event.target.value),
            })),
          (() => {
            const list = filteredPeople();
            if (!list.length) {
              return h('p', { class: 't-dim', text: 'Nobody matches that search.' });
            }
            const box = h('div', { class: 'people-picker' },
              list.map((person) => personRow(person, {
                name: 'ff-rel',
                selected: local.picks.includes(person.name),
                checked: local.picks.includes(person.name),
                onChange: () => toggle(person.name),
              })));
            return box;
          })(),
          h('div', { class: 'relate-actions' },
            h('button', {
              class: 'btn btn-primary', type: 'button',
              disabled: !local.picks.length,
              onclick: search,
            }, icon('search'), 'Show photos'),
            local.picks.length
              ? h('button', {
                  class: 'btn', type: 'button',
                  onclick: () => { local.picks = []; search(); },
                }, 'Clear')
              : null,
            h('span', { class: 'spacer' }),
            h('button', {
              class: 'btn', type: 'button',
              disabled: !photos.length,
              onclick: async () => {
                const folder = await api.pickFolder('export');
                if (!folder) return;
                try {
                  const copied = await api.exportIntersection(local.picks, folder);
                  toast(`Copied ${copied.copied || photos.length} photos`, 'success');
                  api.openPath(folder);
                } catch (error) {
                  toast('Could not copy those photos', 'error', { hint: error.hint });
                }
              },
            }, icon('download'), 'Export these photos')))));

      /* ------------------------------------------------------------ result */
      if (local.unknown && local.unknown.length) {
        children.push(callout({
          kind: 'warning',
          icon: 'alert',
          title: 'Not in the name database',
          message: `${local.unknown.join(', ')} — pick the name as it is spelled on the group.`,
        }));
      }

      if (result) {
        children.push(h('section', { class: 'section' },
          h('div', { class: 'relate-summary' },
            h('div', null,
              h('div', { class: 'relate-count t-num', text: fmt.count(photos.length) }),
              h('div', { class: 't-mute',
                text: result.names && result.names.length > 1
                  ? `photos with ${result.names.join(' and ')}`
                  : (result.names && result.names.length === 1
                      ? `photos of ${result.names[0]}`
                      : 'recorded photos') })),
            h('div', { class: 'relate-meta' },
              h('dl', { class: 'rows' },
                ...(result.names || []).map((name) => h('div', null,
                  h('dt', { text: name }),
                  h('dd', { text: fmt.plural((result.per_person || {})[name] ?? 0, 'photo') }))))))));

        children.push(photos.length
          ? h('section', { class: 'section' },
              h('div', { class: 'section-head' },
                h('h2', { class: 'section-title', text: 'Matching photos' }),
                photos.length > 400
                  ? h('span', { class: 't-mute',
                      text: `Showing the first 400 of ${fmt.count(photos.length)}` })
                  : null),
              photoGrid(photos, { cap: 400, eager: 12 }))
          : emptyState({
              icon: 'image',
              title: 'No photos found',
              note: (result.names || []).length > 1
                ? `${result.names.join(' and ')} never appear in the same photo that has been `
                  + 'named. One of them may be spelled differently than you expect.'
                : 'Their photos may not have been named yet.',
            }));
      } else if (local.searching) {
        children.push(FF.ui.skeleton(0, '200px'));
      }

      /* ----------------------------------------------------- co-occurrence */
      children.push(h('section', { class: 'section' },
        h('div', { class: 'section-head' },
          h('div', null,
            h('h2', { class: 'section-title', text: 'Who appears together' }),
            h('p', { class: 'section-note',
              text: 'Pairs from your library, most shared photos first. '
                  + 'Click a pair to see the photos.' }))),
        local.coOccurrenceError
          ? h('p', { class: 't-dim',
              text: 'The shared-photo map could not be loaded right now. Pairwise search above still works.' })
          : (local.coOccurrence ? pairsView() : FF.ui.skeleton(0, '140px'))));

      mount(host, children);
    }

    render();
    load();
    return { node: host, teardown: () => { disposed = true; onQuery.cancel(); } };
  };
})(window.FF);