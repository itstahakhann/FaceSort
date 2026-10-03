/* ==========================================================================
   FaceFlow — Gallery Export
   A four-step wizard over POST /export_gallery. The steps mirror what the
   user is actually deciding: what goes in, what options apply, wait, open.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, icon } = FF.dom;
  const { api, fmt } = FF;
  const { toast, callout, errorCallout, emptyState } = FF.ui;

  FF.views = FF.views || {};

  const STEPS = ['Choose', 'Options', 'Create', 'Done'];

  FF.views.gallery = function (state) {
    const host = h('div', { class: 'view-body' });
    const local = {
      step: 0,
      mode: 'all',              // all | people
      picks: new Set(),
      people: [],
      thumbnails: true,
      password: '',
      reveal: false,
      path: '',
      building: false,
      progress: null,
      result: null,
      error: null,
      dbUnavailable: false,
    };
    let disposed = false;
    let timer = null;

    /* ---------------------------------------------------------------- data */

    async function loadPeople() {
      try {
        const payload = await api.peopleList();
        local.people = payload.people || [];
        local.dbUnavailable = payload.db === false;
      } catch (error) {
        local.error = error;
      }
      if (!disposed) render();
    }

    /* ------------------------------------------------------------- progress
       The engine builds on a worker thread and reports through
       /gallery_status, so this polls only while a build is actually running
       and stops the moment it finishes. */

    function watchBuild() {
      if (timer) clearInterval(timer);
      timer = setInterval(async () => {
        if (disposed) { clearInterval(timer); timer = null; return; }
        try {
          const status = await api.galleryStatus();
          local.progress = status;
          if (!status.building) {
            clearInterval(timer);
            timer = null;
            local.building = false;
            if (status.result) {
              local.result = status.result;
              local.step = 3;
              FF.logActivity(
                `Gallery built · ${status.result.total_photos} photos`, 'success');
              toast('Gallery ready', 'success');
            } else {
              local.error = new FF.EngineError('The gallery build stopped',
                'The engine reported no result. It may have run out of disk space.');
              local.step = 1;
            }
            render();
          } else {
            render();
          }
        } catch (error) {
          clearInterval(timer);
          timer = null;
          local.building = false;
          local.error = error;
          render();
        }
      }, 500);
    }

    async function build() {
      if (local.mode === 'people' && !local.picks.size) {
        local.error = new FF.EngineError('No people selected',
          'Choose at least one person, or switch to Everyone.');
        render();
        return;
      }
      local.error = null;
      local.building = true;
      local.step = 2;
      local.progress = { done: 0, total: 0, current: '' };
      render();

      try {
        await api.exportGallery({
          output_path: local.path,
          export_mode: local.mode,
          selected_people: [...local.picks],
          password: local.password,
          include_thumbnails: local.thumbnails,
        });
      } catch (error) {
        local.building = false;
        local.error = error;
        local.step = 1;
        render();
        return;
      }
      FF.logActivity('Gallery build started', 'accent');
      watchBuild();
    }

    /* -------------------------------------------------------------- render */

    function stepper() {
      return h('ol', { class: 'wizard-steps', 'aria-label': 'Gallery export steps' },
        STEPS.map((label, index) => h('li', {
          class: `wizard-step ${index === local.step ? 'is-current' : ''} `
               + `${index < local.step ? 'is-done' : ''}`,
          'aria-current': index === local.step ? 'step' : null,
        },
          h('span', { class: 'wizard-dot' },
            index < local.step ? icon('check', 'icon-sm') : String(index + 1)),
          h('span', { class: 'wizard-label', text: label }))));
    }

    function stepChoose() {
      return [
        h('h2', { class: 'section-title', text: 'Create an offline gallery' }),
        h('p', { class: 'section-note', text: 'What would you like to export?' }),
        h('div', { class: 'option-grid' },
          optionCard({
            name: 'all', checked: local.mode === 'all',
            title: 'Everyone',
            note: local.people.length
              ? `${fmt.plural(local.people.length, 'person', 'people')}, every photo recorded`
              : 'Everyone who has been named',
          }),
          optionCard({
            name: 'people', checked: local.mode === 'people',
            title: 'Selected people',
            note: 'Choose exactly who appears',
          })),
        local.mode === 'people'
          ? h('div', { class: 'people-picker', style: { marginTop: 'var(--s5)' } },
              local.people.length
                ? local.people.map((person) => h('label', { class: 'person-row' },
                    h('input', {
                      type: 'checkbox',
                      checked: local.picks.has(person.name),
                      onchange: (event) => {
                        if (event.target.checked) local.picks.add(person.name);
                        else local.picks.delete(person.name);
                        render();
                      },
                    }),
                    h('span', { class: 'person-row-name truncate', text: person.name }),
                    h('span', { class: 'person-row-count t-num', text: fmt.count(person.photos) })))
                : h('p', { class: 't-dim', text: 'Nobody has been named yet.' }))
          : null,
      ];
    }

    function optionCard(config) {
      return h('label', { class: 'option' },
        h('input', {
          type: 'radio', name: 'ff-gal-mode', value: config.name,
          checked: config.checked,
          onchange: () => { local.mode = config.name; render(); },
        }),
        h('span', null,
          h('span', { class: 'option-title', text: config.title }),
          h('span', { class: 'option-note', text: config.note })));
    }

    function stepOptions() {
      const pathInput = h('input', {
        class: 'input input-mono', type: 'text', id: 'ff-gal-path',
        placeholder: 'Choose a folder', value: local.path, spellcheck: 'false',
        autocomplete: 'off',
      });
      pathInput.addEventListener('input', () => { local.path = pathInput.value.trim(); });

      return [
        h('h2', { class: 'section-title', text: 'Options' }),
        h('p', { class: 'section-note',
          text: 'A gallery is a folder of one HTML page, a stylesheet, a script and your '
              + 'photos. It opens in any browser with no network, so it can be zipped, '
              + 'emailed, or put on a USB stick.' }),

        h('label', { class: 'option', style: { marginTop: 'var(--s5)' } },
          h('input', {
            type: 'checkbox', checked: local.thumbnails,
            onchange: (event) => { local.thumbnails = event.target.checked; },
          }),
          h('span', null,
            h('span', { class: 'option-title', text: 'Generate thumbnails' }),
            h('span', { class: 'option-note',
              text: 'Small copies for the grid, so the gallery opens quickly. Recommended.' }))),

        h('label', { class: 'option', style: { marginTop: 'var(--s2)' } },
          h('input', {
            type: 'checkbox', checked: Boolean(local.password),
            onchange: (event) => {
              local.password = event.target.checked ? (local.password || 'faceflow') : '';
              render();
            },
          }),
          h('span', null,
            h('span', { class: 'option-title', text: 'Password protect the gallery' }),
            h('span', { class: 'option-note',
              text: 'Adds a password screen in front of the page.' }))),

        local.password
          ? h('div', { class: 'field', style: { marginTop: 'var(--s4)', maxWidth: '360px' } },
              h('label', { class: 'label', for: 'ff-gal-pass', text: 'Gallery password' }),
              h('div', { class: 'input-row' },
                h('input', {
                  class: 'input', id: 'ff-gal-pass', type: local.reveal ? 'text' : 'password',
                  value: local.password, autocomplete: 'new-password',
                  oninput: (event) => { local.password = event.target.value; },
                }),
                h('button', {
                  class: 'btn btn-icon', type: 'button',
                  'aria-label': local.reveal ? 'Hide password' : 'Show password',
                  title: local.reveal ? 'Hide password' : 'Show password',
                  onclick: () => { local.reveal = !local.reveal; render(); },
                }, icon('eye'))),
              // Say plainly what the password is, because it is not what most
              // people assume.
              callout({
                kind: 'warning',
                icon: 'alert',
                title: 'This is a gate, not encryption',
                message: 'The password check runs in the browser and its digest is stored '
                      + 'inside the HTML file. Anyone technical can read the photos directly. '
                      + 'It deters casual browsing; it does not protect against someone '
                      + 'determined. Keep the photos somewhere private if that matters.',
              }))
          : null,

        h('div', { class: 'field', style: { marginTop: 'var(--s5)', maxWidth: '560px' } },
          h('label', { class: 'label', for: 'ff-gal-path', text: 'Save the gallery to' }),
          h('div', { class: 'input-row' },
            pathInput,
            h('button', {
              class: 'btn btn-icon', type: 'button',
              'aria-label': 'Choose output folder',
              onclick: async () => {
                const folder = await api.pickFolder('gallery');
                if (!folder) return;
                // The generator owns the folder it writes, so append a name
                // rather than filling the folder the user pointed at.
                local.path = fmt.joinPath(folder, 'FaceFlow Gallery');
                render();
              },
            }, icon('folder-open')))),
      ];
    }

    function stepCreate() {
      const status = local.progress || { done: 0, total: 0, current: '' };
      const total = status.total || 0;
      const done = status.done || 0;
      const pct = total ? Math.min(100, (done / total) * 100) : 0;
      return [
        h('h2', { class: 'section-title', text: 'Creating your gallery' }),
        h('p', { class: 'section-note',
          text: 'Copying photos and building the page. This happens on this computer.' }),
        h('div', { class: 'scan-count t-num', style: { marginTop: 'var(--s5)' } },
          fmt.count(done),
          h('span', { class: 'scan-count-total', text: total ? ` / ${fmt.count(total)}` : '' })),
        h('div', { class: 'progress progress-lg' },
          h('div', { class: 'progress-bar', style: { width: `${pct}%` } })),
        h('p', { class: 't-mute',
          text: local.thumbnails ? 'Creating thumbnails and copying photos…'
                                  : 'Copying photos…' }),
      ];
    }

    function stepDone() {
      const result = local.result;
      if (!result) return [h('p', { class: 't-dim', text: 'Preparing…' })];
      const mb = fmt.bytes(result.bytes_written || 0);
      return [
        h('div', { class: 'done-mark' }, icon('check', 'icon-lg')),
        h('h2', { class: 'section-title', text: 'Gallery ready' }),
        h('p', { class: 'section-note', text: 'Saved in' }),
        h('p', { class: 'input-mono gal-path', text: result.output_path }),
        h('dl', { class: 'rows rows-wide' },
          row('Photos', fmt.count(result.total_photos)),
          row('People', fmt.plural((result.people || []).length, 'person', 'people')),
          row('Thumbnails', fmt.count(result.thumbnails || 0)),
          row('Size', mb),
          result.missing ? row('Missing from disk', fmt.count(result.missing)) : null,
          result.skipped ? row('Skipped (unreadable)', fmt.count(result.skipped)) : null),
        (result.missing || result.skipped)
          ? h('p', { class: 'field-note',
              text: 'Some recorded photos were no longer on disk or could not be read. '
                  + 'They were skipped rather than failing the whole build.' })
          : null,
        h('div', { class: 'scan-launch' },
          h('button', {
            class: 'btn btn-primary', type: 'button',
            onclick: () => api.openPath(fmt.joinPath(result.output_path, 'index.html')),
          }, icon('eye'), 'Open gallery'),
          h('button', {
            class: 'btn', type: 'button',
            onclick: () => api.openPath(result.output_path),
          }, icon('folder-open'), 'Open folder'),
          h('button', {
            class: 'btn', type: 'button',
            onclick: () => { local.step = 0; local.result = null; render(); },
          }, 'Make another')),
      ];
    }

    const row = (label, value) => h('div', null,
      h('dt', { text: label }), h('dd', { text: value }));

    function render() {
      if (disposed) return;
      const children = [
        h('div', { class: 'page-head' },
          h('h1', { class: 'page-title', text: 'Gallery export' }),
          h('p', { class: 'page-sub',
            text: 'Build a website of your photos that works with no internet and no '
                + 'server — and keeps working in ten years.' })),
        stepper(),
      ];

      if (local.dbUnavailable && local.step === 0) {
        children.push(callout({
          kind: 'warning',
          icon: 'alert',
          title: 'No named people yet',
          message: 'A gallery is built from the groups you have named, so there is nothing '
                + 'to export until you name some.',
          action: h('button', {
            class: 'btn btn-sm', type: 'button', onclick: () => FF.go('photos'),
          }, 'Review groups'),
        }));
      }

      if (local.error) children.push(errorCallout(local.error));

      children.push(h('section', { class: 'panel' },
        h('div', { class: 'panel-body' },
          local.step === 0 ? stepChoose()
          : local.step === 1 ? stepOptions()
          : local.step === 2 ? stepCreate()
          : stepDone())));

      /* -------------------------------------------------------- navigation */
      if (local.step < 2) {
        children.push(h('div', { class: 'scan-launch' },
          local.step > 0
            ? h('button', {
                class: 'btn', type: 'button',
                onclick: () => { local.step -= 1; render(); },
              }, icon('arrow-left'), 'Back')
            : null,
          local.step === 1
            ? h('button', {
                class: 'btn btn-primary', type: 'button',
                disabled: !local.path,
                title: local.path ? '' : 'Choose where to save the gallery',
                onclick: () => { local.step = 2; build(); },
              }, icon('camera'), 'Create gallery')
            : h('button', {
                class: 'btn btn-primary', type: 'button',
                onclick: () => { local.step = 1; render(); },
              }, 'Continue'),
            h('button', {
              class: 'btn', type: 'button', onclick: () => FF.go('overview'),
            }, 'Cancel')));
      }

      mount(host, children);
    }

    render();
    loadPeople();
    return {
      node: host,
      teardown: () => { disposed = true; if (timer) { clearInterval(timer); timer = null; } },
    };
  };
})(window.FF);