/* ==========================================================================
   FaceFlow — DOM helpers
   --------------------------------------------------------------------------
   The renderer is loaded over file://, where ES modules are blocked by CORS,
   so modules here are classic scripts attached to one global namespace.

   Text always goes in through textContent, never innerHTML. Person names and
   file paths come from the user's disk, and a name like
   `"><img onerror=...>` must render as those characters, not as markup.
   ========================================================================== */
window.FF = window.FF || {};

(function (FF) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  /**
   * Create an element.
   *
   *   h('div', { class: 'card' }, 'text')
   *   h('button', { onclick: fn, 'aria-label': 'Close' }, h('span', 'x'))
   *
   * Props: `class`, `id`, `style` (object), `dataset` (object), `on*` for
   * events, everything else set as an attribute. A `false`/`null`/`undefined`
   * attribute is skipped so optional attributes can be written inline.
   */
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    if (props) {
      for (const key of Object.keys(props)) {
        const value = props[key];
        if (value === null || value === undefined || value === false) continue;
        if (key === 'class') el.className = value;
        else if (key === 'text') el.textContent = value;
        else if (key === 'html') el.innerHTML = value;   // only for trusted markup
        else if (key === 'dataset') Object.assign(el.dataset, value);
        else if (key === 'style' && typeof value === 'object') applyStyle(el, value);
        else if (key.startsWith('on') && typeof value === 'function') {
          el.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value === true) el.setAttribute(key, '');
        else el.setAttribute(key, value);
      }
    }
    append(el, children);
    return el;
  }

  /**
   * Apply a style object to an element.
   *
   * Custom properties (`--i`, `--weight`, …) MUST go through setProperty.
   * Object.assign(el.style, {'--i': 1}) does nothing at all — the CSSOM
   * ignores unknown properties — so stagger delays and bar widths silently
   * fell back to their defaults.
   */
  function applyStyle(el, styles) {
    for (const name of Object.keys(styles)) {
      const value = styles[name];
      if (value === null || value === undefined) continue;
      if (name.startsWith('--')) el.style.setProperty(name, String(value));
      else el.style[name] = value;
    }
  }

  function append(parent, children) {
    for (const child of children) {
      if (child === null || child === undefined || child === false) continue;
      if (Array.isArray(child)) append(parent, child);
      else if (child instanceof Node) parent.appendChild(child);
      else parent.appendChild(document.createTextNode(String(child)));
    }
    return parent;
  }

  /** Replace a node's children in one operation. */
  function mount(parent, ...children) {
    parent.replaceChildren();
    append(parent, children);
    return parent;
  }

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => [...(root || document).querySelectorAll(selector)];

  function clear(node) { if (node) node.replaceChildren(); return node; }

  /** An <svg><use> reference into the inline sprite. */
  function icon(name, extraClass) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', `icon ${extraClass || ''}`.trim());
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS(SVG_NS, 'use');
    use.setAttribute('href', `#i-${name}`);
    svg.appendChild(use);
    return svg;
  }

  /**
   * Trailing-edge debounce. Search boxes fire per keystroke; without this a
   * 1,000-cluster library re-filters on every character.
   */
  function debounce(fn, wait) {
    let timer = null;
    const wrapped = (...args) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { timer = null; fn(...args); }, wait || 140);
    };
    wrapped.cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
    wrapped.flush = (...args) => { if (timer) { clearTimeout(timer); timer = null; } fn(...args); };
    return wrapped;
  }

  /** Set --i on each child so a CSS stagger can read it. */
  function stagger(container, items, cap) {
    const limit = cap || 18;
    items.forEach((item, index) => {
      item.style.setProperty('--i', String(Math.min(index, limit)));
    });
    return container;
  }

  /** Focus trap + restore, used by dialogs and the lightbox. */
  const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),' +
    'select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

  function trapFocus(container, event) {
    if (event.key !== 'Tab') return;
    const nodes = $$(FOCUSABLE, container).filter((node) => node.offsetParent !== null);
    if (!nodes.length) return;
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  /** run when motion is allowed; no-op when the user asked for less of it. */
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  FF.dom = {
    h, mount, append, $, $$, clear, icon, debounce, stagger, trapFocus, SVG_NS,
    get reducedMotion() { return reducedMotion.matches; },
    on(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      return () => target.removeEventListener(type, handler, options);
    },
  };
})(window.FF);