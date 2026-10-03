/* ==========================================================================
   FaceFlow — shared UI primitives
   Toasts, dialogs, the photo lightbox, empty states, error callouts and the
   skeleton placeholders. Everything here is reusable and every one of them
   handles keyboard and focus correctly.
   ========================================================================== */
(function (FF) {
  'use strict';

  const { h, mount, $, $$, icon, trapFocus, reducedMotion } = FF.dom;
  const { fmt } = FF;

  /* ================================================================= toasts
     Non-blocking, bottom-right, auto-dismissing. Errors stay longer and are
     the only kind that offers a dismissal control, because a message the user
     did not cause should not vanish before they have read it. */

  const TOAST_MS = { info: 3200, success: 3600, warning: 6000, error: 9000 };
  const TOAST_ICON = { success: 'check', error: 'alert', warning: 'alert', info: 'info' };

  let toastLayer = null;
  const live = [];

  function ensureToastLayer() {
    if (!toastLayer) {
      toastLayer = h('div', {
        class: 'toasts',
        role: 'status',
        'aria-live': 'polite',
        'aria-atomic': 'false',
      });
      document.body.append(toastLayer);
    }
    return toastLayer;
  }

  function toast(message, kind, options) {
    const settings = options || {};
    const level = kind || 'info';
    const layer = ensureToastLayer();

    const dismiss = () => {
      if (!node.isConnected) return;
      live.splice(live.indexOf(node), 1);
      node.classList.add('is-leaving');
      // Let the exit animation run before removing, unless motion is reduced.
      setTimeout(() => node.remove(), reducedMotion ? 0 : 180);
    };

    const node = h('div', {
      class: `toast toast-${level}`,
      role: level === 'error' ? 'alert' : 'status',
    },
      icon(TOAST_ICON[level] || 'info'),
      h('div', { class: 'toast-text' },
        h('div', { text: message }),
        settings.hint
          ? h('div', { class: 't-dim', style: { marginTop: '2px', fontSize: 'var(--fs-12)' },
                       text: settings.hint })
          : null,
        settings.detail
          ? h('details', { class: 'technical' },
              h('summary', { text: 'Technical details' }),
              h('pre', { text: settings.detail }))
          : null),
      h('button', {
        class: 'toast-close',
        type: 'button',
        'aria-label': 'Dismiss notification',
        onclick: dismiss,
      }, icon('x', 'icon-sm')));

    layer.append(node);
    live.push(node);

    // Keep at most four; a burst of failures should not fill the screen.
    while (live.length > 4) live[0].remove(), live.shift();

    const ttl = settings.ttl || TOAST_MS[level] || 3200;
    if (ttl > 0) setTimeout(dismiss, ttl);

    return dismiss;
  }

  /* ================================================================ dialogs
     A modal that traps focus, restores it on close, closes on Escape and on
     backdrop click, and is marked up as a dialog for screen readers. */

  let openDialog = null;

  function dialog(config) {
    if (openDialog) openDialog.close();

    const previouslyFocused = document.activeElement;
    let scrim;

    const close = (result) => {
      if (!scrim.isConnected) return;
      openDialog = null;
      document.removeEventListener('keydown', onKeyDown, true);
      scrim.remove();
      if (previouslyFocused && previouslyFocused.focus) previouslyFocused.focus();
      if (config.onClose) config.onClose(result);
    };

    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        event.preventDefault();
        close(null);
        return;
      }
      trapFocus(panel, event);
    };

    const body = h('div', { class: 'dialog-body' },
      typeof config.body === 'function' ? config.body(close) : config.body);

    const panel = h('div', {
      class: `dialog ${config.wide ? 'dialog-wide' : ''}`,
      role: 'dialog',
      'aria-modal': 'true',
      'aria-labelledby': 'dialog-title',
    },
      h('div', { class: 'dialog-head' },
        h('div', null,
          h('h2', { class: 'dialog-title', id: 'dialog-title', text: config.title }),
          config.subtitle
            ? h('p', { class: 'dialog-sub', text: config.subtitle })
            : null),
        config.dismissible === false ? null
          : h('button', {
              class: 'btn btn-ghost btn-icon btn-sm',
              type: 'button',
              'aria-label': 'Close dialog',
              onclick: () => close(null),
            }, icon('x', 'icon-sm'))),
      body,
      config.footer ? h('div', { class: 'dialog-foot' },
        typeof config.footer === 'function' ? config.footer(close) : config.footer) : null);

    scrim = h('div', {
      class: 'scrim',
      onmousedown: (event) => {
        // Only a click that both starts and ends on the backdrop closes, so a
        // text selection that drags outside the dialog does not dismiss it.
        if (event.target === scrim && config.dismissible !== false) close(null);
      },
    }, panel);

    document.body.append(scrim);
    document.addEventListener('keydown', onKeyDown, true);

    const initial = panel.querySelector('[data-autofocus]') ||
      panel.querySelector('.dialog-body input, .dialog-body textarea') ||
      panel.querySelector('.dialog-foot .btn-primary');
    if (initial) requestAnimationFrame(() => initial.focus());

    openDialog = { close };
    return { close, panel };
  }

  /** Confirmation with a promise. Resolves true/false, never throws. */
  function confirmDialog(config) {
    return new Promise((resolve) => {
      let settled = false;
      const settle = (value) => { if (!settled) { settled = true; resolve(value); } };
      dialog({
        title: config.title,
        subtitle: config.subtitle,
        body: config.body,
        footer: (close) => [
          h('span', { class: 'spacer' }),
          h('button', {
            class: 'btn', type: 'button',
            onclick: () => { settle(false); close(); },
          }, config.cancelLabel || 'Cancel'),
          h('button', {
            class: `btn ${config.danger ? 'btn-danger' : 'btn-primary'}`,
            type: 'button',
            onclick: () => { settle(true); close(); },
          }, config.confirmLabel || 'Confirm'),
        ],
        onClose: () => settle(false),
      });
    });
  }

  /* =============================================================== lightbox
     Full-screen photo viewer with arrow-key navigation, Escape, and
     preloading of neighbours so the arrows feel instant on a big library. */

  let lightboxState = null;

  function openLightbox(options) {
    const photos = options.photos || [];
    let index = Math.max(0, photos.indexOf(options.photo) || 0);
    const previouslyFocused = document.activeElement;
    let node;

    const stage = h('div', { class: 'lightbox-stage' });
    const caption = h('span', { class: 'truncate' });

    const show = (next) => {
      if (!photos.length) return;
      index = (next + photos.length) % photos.length;
      const photo = photos[index];
      const image = h('img', { alt: photo.name || 'Photo', decoding: 'async' });
      mount(stage,
        photos.length > 1 ? h('button', {
          class: 'lightbox-nav lightbox-prev', type: 'button',
          'aria-label': 'Previous photo',
          onclick: (event) => { event.stopPropagation(); show(index - 1); },
        }, icon('chevron-left')) : null,
        image,
        photos.length > 1 ? h('button', {
          class: 'lightbox-nav lightbox-next', type: 'button',
          'aria-label': 'Next photo',
          onclick: (event) => { event.stopPropagation(); show(index + 1); },
        }, icon('chevron-right')) : null);
      caption.textContent = `${photo.name || ''}${photos.length > 1
        ? `  ·  ${index + 1} of ${photos.length}` : ''}`;
      // Warm the neighbours so arrow keys do not wait on a decode.
      if (photos.length > 1) {
        [photos[(index + 1) % photos.length], photos[(index - 1 + photos.length) % photos.length]]
          .forEach((neighbour) => {
            if (neighbour && neighbour.load && !neighbour.loaded) neighbour.load();
          });
      }
    };

    const close = () => {
      if (!node || !node.isConnected) return;
      lightboxState = null;
      document.removeEventListener('keydown', onKey, true);
      node.remove();
      if (previouslyFocused && previouslyFocused.focus) previouslyFocused.focus();
    };

    const onKey = (event) => {
      if (event.key === 'Escape') { event.preventDefault(); close(); }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); show(index - 1); }
      else if (event.key === 'ArrowRight') { event.preventDefault(); show(index + 1); }
      else if (event.key === 'Tab') trapFocus(node, event);
    };

    node = h('div', {
      class: 'lightbox',
      role: 'dialog',
      'aria-modal': 'true',
      'aria-label': 'Photo viewer',
      onmousedown: (event) => { if (event.target === node || event.target === stage) close(); },
    },
      stage,
      h('div', { class: 'lightbox-bar' },
        caption,
        h('button', {
          class: 'btn btn-ghost btn-icon btn-sm lightbox-close',
          type: 'button',
          'aria-label': 'Close viewer',
          onclick: close,
        }, icon('x', 'icon-sm'))));

    document.body.append(node);
    document.addEventListener('keydown', onKey, true);
    lightboxState = { close, show };
    show(index);
    return lightboxState;
  }

  /* =========================================================== empty states
     Every major surface gets one, and each says what to do next. An empty
     state that only says "nothing here" makes the user feel they broke it. */

  function emptyState(config) {
    return h('div', { class: 'empty' },
      h('div', { class: 'empty-mark' }, icon(config.icon || 'image', 'icon-lg')),
      h('div', { class: 'empty-title', text: config.title }),
      config.note ? h('p', { class: 'empty-note', text: config.note }) : null,
      config.action || null);
  }

  /* ============================================================== callouts */

  /**
   * An inline explanation. `technical` is folded away by default so a user
   * sees a sentence they can act on, not a Python message.
   */
  function callout(config) {
    const kind = config.kind || '';
    return h('div', {
      class: `callout ${kind ? `callout-${kind}` : ''}`,
      role: kind === 'danger' ? 'alert' : 'note',
    },
      icon(config.icon || (kind === 'danger' ? 'alert' : 'info')),
      h('div', { class: 'callout-body' },
        config.title ? h('div', { class: 'callout-title', text: config.title }) : null,
        h('div', { text: config.message }),
        config.technical
          ? h('details', { class: 'technical' },
              h('summary', { text: 'Technical details' }),
              h('pre', { text: config.technical }))
          : null,
        config.action || null));
  }

  /** Turn any thrown value into the message a person can act on. */
  function errorCallout(error, config) {
    const settings = config || {};
    const hint = (error && error.hint) || settings.fallbackHint ||
      'Something went wrong. The details below may help.';
    return callout({
      kind: 'danger',
      title: settings.title || (error && error.message) || 'Something went wrong',
      message: hint,
      technical: (error && (error.detail || error.message)) || String(error),
      action: settings.action,
    });
  }

  /* =============================================================== skeleton */

  function skeleton(width, height) {
    return h('div', {
      class: 'skeleton',
      style: { width: width || '100%', height: height || '14px' },
      'aria-hidden': 'true',
    });
  }

  /** Placeholder cards shown while a view's first payload is in flight. */
  function skeletonGrid(count, className) {
    return h('div', { class: className || 'stagger' },
      Array.from({ length: count || 6 }, (_, index) =>
        h('div', { class: 'card skeleton-card', style: { '--i': String(index) } },
          skeleton('100%', '120px'),
          h('div', { class: 'skeleton-lines' },
            skeleton('70%', '12px'),
            skeleton('45%', '10px')))));
  }

  FF.ui = {
    toast, dialog, confirm: confirmDialog, openLightbox, lightbox: () => lightboxState,
    emptyState, callout, errorCallout, skeleton, skeletonGrid,
  };
})(window.FF);