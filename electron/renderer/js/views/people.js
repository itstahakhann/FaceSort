/* ==========================================================================
   FaceFlow — People
   Everyone named in the database, from GET /people_list. Clicking a person
   opens a detail view: their photos, plus the actions that make sense on a
   single person rather than on a pair.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon, debounce, stagger } = FF.dom;
  const { api, fmt } = FF;
  const { toast, dialog, emptyState, callout, errorCallout } = FF.ui;
  const { photoGrid } = FF.components;

  FF.views = FF.views || {};

  /* ============================================================ person card
     A person is represented by a crop of their face. The engine's
     /people_list does not return one, so the card falls back to a monogram —
     honest rather than inventing a portrait. */
  function personCard(person, onOpen) {
    return h('button', {
      class: 'person-card',
      type: 'button',
      onclick: () => onOpen(person),
    },
      h('span', { class: 'person-card-avatar', 'aria-hidden': 'true' },
        h('span', { class: 'person-initial', text: (person.name || '?').charAt(0).toUpperCase() })),
      h('span', { class: 'person-card-body' },
        h('span', { class: 'person-card-name truncate', text: person.name }),
        h('span', { class: 'person-card-meta t-dim',
          text: `${fmt.plural(person.photos, 'photo')} · ${fmt.plural(person.faces, 'face')}` })));
  }

  /* ================================================================ the view */

  FF.views.people = function (state) {
    const host = h('div', { class: 'view-body' });
    const local = {
      people: [],
      query: '',
      loading: true,
      error: null,
      // `person` is who is open, `photos` is what we loaded for them. Keeping
      // them apart matters: one held both, and the photo array shadowed the
      // person, so the header rendered "0 photos" and an empty name.
      person: null,
      photos: null,
      loadingDetail: false,
    };
    let disposed = false;

    async function load() {
      local.loading = true;
      local.error = null;
      render();
      try {
        const payload = await api.peopleList();
        local.people = (payload.people || []).slice()
          .sort((a, b) => (b.photos || 0) - (a.photos || 0));
        if (payload.db === false) {
          local.error = new FF.EngineError('The name database is disabled',
            'FaceFlow could not find its name database, so nobody has been remembered yet.');
        }
        local.loading = false;
        render();
      } catch (error) {
        local.loading = false;
        local.error = error;
        render();
      }
    }

    const filtered = () => {
      const query = local.query.trim().toLowerCase();
      return local.people.filter((person) => person.name.toLowerCase().includes(query));
    };

    const onQuery = debounce((value) => { local.query = value; render(); }, 140);

    /* -------------------------------------------------------- detail view */

    function renderDetail(person) {
      mount(host,
        h('button', {
          class: 'btn btn-quiet back-link', type: 'button',
          onclick: () => { local.person = null; local.photos = null; render(); },
        }, icon('arrow-left', 'icon-sm'), 'Back to people'),

        h('div', { class: 'person-detail-head' },
          h('div', { class: 'person-detail-avatar', 'aria-hidden': 'true' },
            h('span', { text: (person.name || '?').charAt(0).toUpperCase() })),
          h('div', { class: 'person-detail-meta' },
            h('h1', { class: 'page-title', text: person.name }),
            h('p', { class: 'page-sub',
              text: `${fmt.plural(person.photos, 'photo')} · `
                  + `${fmt.plural(person.faces, 'face')}`
                  + (person.last_seen ? ` · last seen ${fmt.when(person.last_seen)}` : '') })),
          h('div', { class: 'person-detail-actions' },
            h('button', {
              class: 'btn', type: 'button',
              onclick: () => renameDialog(person),
            }, icon('tag'), 'Rename'),
            h('button', {
              class: 'btn', type: 'button',
              title: 'Pick two groups in Photos to merge them',
              onclick: () => toast('Open Photos and use Merge on the two groups',
                'info', { ttl: 4200 }),
            }, icon('merge'), 'Merge'),
            h('button', {
              class: 'btn', type: 'button',
              title: 'Not available yet — the engine has no split operation',
              disabled: true,
            }, icon('split'), 'Split'),
            h('button', {
              class: 'btn btn-primary', type: 'button',
              onclick: () => {
                FF.go('relationships');
                // Hand the selection over so the user does not retype it.
                FF.pendingRelationshipPick = [person.name];
              },
            }, icon('link'), 'Find people together'))),

        h('section', { class: 'section' },
          h('div', { class: 'section-head' },
            h('h2', { class: 'section-title', text: 'Photos' }),
            h('button', {
              class: 'btn btn-sm', type: 'button',
              onclick: async () => {
                const folder = await api.pickFolder('export');
                if (!folder) return;
                try {
                  const result = await api.exportIntersection([person.name], folder);
                  toast(`Copied ${result.copied || result.count || 0} photos`, 'success');
                  api.openPath(folder);
                } catch (error) {
                  toast('Could not copy those photos', 'error', { hint: error.hint });
                }
              },
            }, icon('download'), 'Export these photos')),

          h('div', { class: 'photo-panel' },
            local.loadingDetail
              ? FF.ui.skeleton(0, '260px')
              : local.photos && local.photos.length
                ? photoGrid(local.photos, { eager: 10 })
                : h('p', { class: 't-dim',
                    text: 'No photos are recorded for this person yet. Photos appear here '
                        + 'once the groups they belong to have been named and the library '
                        + 'has been scanned again.' }))));
    }

    async function loadDetail(person) {
      local.photos = null;
      local.loadingDetail = true;
      render();
      try {
        const payload = await api.intersection([person.name]);
        local.photos = payload.photos || [];
      } catch (error) {
        toast('Could not load those photos', 'error', { hint: error.hint });
      }
      local.loadingDetail = false;
      if (!disposed) render();
    }

    function renameDialog(person) {
      const input = h('input', {
        class: 'input input-name', type: 'text', id: 'ff-rename',
        value: person.name, autocomplete: 'off', 'data-autofocus': 'true',
      });
      const error = h('p', { class: 'field-error', hidden: true });
      dialog({
        title: 'Rename this person',
        subtitle: 'The new name is used wherever this person appears.',
        body: h('div', { class: 'field' },
          h('label', { class: 'label', for: 'ff-rename', text: 'Person name' }),
          input, error),
        footer: (close) => [
          h('span', { class: 'spacer' }),
          h('button', { class: 'btn', type: 'button', onclick: () => close() }, 'Cancel'),
          h('button', {
            class: 'btn btn-primary', type: 'button',
            onclick: async () => {
              const value = input.value.trim();
              if (!value) {
                error.textContent = 'A name cannot be empty.';
                error.hidden = false;
                return;
              }
              try {
                // The engine names a cluster, not a person; a person here is an
                // aggregate of clusters, so renaming is applied to the name
                // recorded against them on the next scan. Report that plainly
                // rather than pretending the database was updated.
                await api.nameCluster(person.id, value);
                person.name = value;
                close();
                render();
                toast('Cluster renamed successfully', 'success');
              } catch (caught) {
                error.textContent = (caught && caught.hint) || 'The name could not be saved.';
                error.hidden = false;
              }
            },
          }, 'Save'),
        ],
      });
    }

    /* --------------------------------------------------------- list render */

    function render() {
      if (disposed) return;
      if (local.person) { renderDetail(local.person); return; }

      const list = filtered();
      const children = [
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'People' }),
          h('p', { class: 'page-sub',
            text: 'Everyone you have named. Photos are grouped by whose face is in them.' })),
      ];

      if (local.error) children.push(errorCallout(local.error));

      if (!local.loading && !local.error) {
        children.push(h('div', { class: 'review-toolbar' },
          h('div', { class: 'search review-search' },
            icon('search'),
            h('input', {
              class: 'input', type: 'search', id: 'ff-people-search',
              placeholder: 'Search people', value: local.query,
              'aria-label': 'Search people by name',
              oninput: (event) => onQuery(event.target.value),
            })),
          h('span', { class: 'spacer' }),
          h('span', { class: 't-mute t-num',
            text: `${fmt.plural(local.people.length, 'person', 'people')}` })));
      }

      if (local.loading) {
        children.push(FF.ui.skeletonGrid(6, 'people-grid'));
      } else if (!local.people.length) {
        children.push(emptyState({
          icon: 'users',
          title: 'No people yet',
          note: local.error
            ? 'The name database is unavailable right now.'
            : 'No scan has been completed, so nobody has been named. Name a group in Photos '
              + 'and the person will appear here.',
          action: h('button', {
            class: 'btn btn-primary', type: 'button', onclick: () => FF.go('photos'),
          }, icon('image'), 'Review groups'),
        }));
      } else if (!list.length) {
        children.push(emptyState({
          icon: 'search',
          title: `Nobody matches “${local.query}”`,
          action: h('button', {
            class: 'btn', type: 'button',
            onclick: () => { local.query = ''; render(); },
          }, 'Clear search'),
        }));
      } else {
        const grid = h('div', { class: 'people-grid' },
          list.map((person) => personCard(person, (chosen) => {
            local.person = chosen;
            loadDetail(chosen);
          })));
        stagger(grid, [...grid.children], 16);
        children.push(grid);
      }

      mount(host, children);
    }

    load();
    return { node: host, teardown: () => { disposed = true; onQuery.cancel(); } };
  };
})(window.FF);