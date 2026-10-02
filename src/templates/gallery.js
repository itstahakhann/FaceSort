/* ==========================================================================
   FaceSort gallery — viewer behaviour
   --------------------------------------------------------------------------
   Vanilla JS, no build step, no dependencies. The photo data arrives as
   window.__GALLERY__ (a JSON array emitted by Jinja2).

   Sections:
     1. theme        2. people grid      3. one person's photos
     4. lightbox     5. password gate
   ========================================================================== */
(function () {
  'use strict';

  var DATA = window.__GALLERY__ || [];
  var THEME_KEY = 'facesort.gallery.theme';

  var peopleView = document.getElementById('view-people');
  var personView = document.getElementById('view-person');
  var grid = document.getElementById('photo-grid');
  var filter = document.getElementById('filter');

  /* ========================================================== 1. theme */
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* private mode */ }
  }

  function initTheme() {
    var stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch (e) { /* ignore */ }
    // Dark is the gallery's default look; only override it if the viewer has
    // an explicit preference (or asks for light and has never chosen).
    var preferred = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light' : 'dark';
    applyTheme(stored || preferred);
    var toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
      });
    }
  }

  /* ==================================================== 2. people grid */
  function initPeople() {
    var grid = document.getElementById('people-grid');
    var empty = document.getElementById('people-empty');
    if (!grid) return;

    grid.addEventListener('click', function (event) {
      var card = event.target.closest('.person');
      if (!card) return;
      openPerson(card.dataset.index);
    });

    if (filter) {
      filter.addEventListener('input', function () {
        var query = filter.value.trim().toLowerCase();
        var shown = 0;
        Array.prototype.forEach.call(grid.children, function (card) {
          var name = (card.dataset.person || '').toLowerCase();
          var match = !query || name.indexOf(query) > -1;
          card.hidden = !match;
          if (match) shown += 1;
        });
        if (empty) empty.hidden = shown !== 0;
      });
    }
  }

  /* ================================================ 3. one person's photos */
  var current = null;   // {name, photos}
  var currentIndex = 0;

  function openPerson(index) {
    var person = DATA[index];
    if (!person) return;
    current = person;
    currentIndex = 0;

    document.getElementById('person-name').textContent = person.name;
    document.getElementById('person-count').textContent =
      person.count + ' photo' + (person.count === 1 ? '' : 's');

    grid.replaceChildren();
    person.photos.forEach(function (photo, position) {
      var tile = document.createElement('button');
      tile.className = 'tile';
      tile.type = 'button';
      tile.style.setProperty('--delay', Math.min(position, 24) * 22 + 'ms');
      tile.setAttribute('aria-label', photo.name);

      var image = document.createElement('img');
      // Thumbnails keep the grid light; the lightbox loads the full frame.
      image.src = photo.thumb || photo.src;
      image.alt = photo.name;
      image.loading = 'lazy';
      image.decoding = 'async';

      tile.appendChild(image);
      tile.addEventListener('click', function () { openLightbox(position); });
      grid.appendChild(tile);
    });

    peopleView.hidden = true;
    personView.hidden = false;
    document.getElementById('stage').scrollTop = 0;
    if (filter) filter.value = '';
  }

  function closePerson() {
    personView.hidden = true;
    peopleView.hidden = false;
  }

  /* ========================================================= 4. lightbox */
  var lightbox = document.getElementById('lightbox');
  var lbImage = document.getElementById('lb-img');
  var lbCaption = document.getElementById('lb-caption');
  var lastFocused = null;

  function show(position) {
    if (!current || !current.photos.length) return;
    currentIndex = (position + current.photos.length) % current.photos.length;
    var photo = current.photos[currentIndex];

    // Preload the neighbours so arrow keys feel instant on a big library.
    var next = current.photos[(currentIndex + 1) % current.photos.length];
    var prev = current.photos[(currentIndex - 1 + current.photos.length) % current.photos.length];
    [next, prev].forEach(function (neighbour) {
      if (neighbour && neighbour.src) { var img = new Image(); img.src = neighbour.src; }
    });

    lbImage.src = photo.src;
    lbImage.alt = photo.name;
    lbCaption.textContent =
      current.name + ' · ' + (currentIndex + 1) + ' of ' + current.photos.length;
    document.body.style.overflow = 'hidden';
  }

  function openLightbox(position) {
    lastFocused = document.activeElement;
    lightbox.hidden = false;
    show(position);
    var close = document.getElementById('lb-close');
    if (close) close.focus();
  }

  function closeLightbox() {
    if (lightbox.hidden) return;
    lightbox.hidden = true;
    lbImage.removeAttribute('src');
    document.body.style.overflow = '';
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  function initLightbox() {
    document.getElementById('lb-close').addEventListener('click', closeLightbox);
    document.getElementById('lb-prev').addEventListener('click', function (e) {
      e.stopPropagation(); show(currentIndex - 1);
    });
    document.getElementById('lb-next').addEventListener('click', function (e) {
      e.stopPropagation(); show(currentIndex + 1);
    });
    lightbox.addEventListener('click', function (event) {
      if (event.target === lightbox) closeLightbox();
    });

    document.addEventListener('keydown', function (event) {
      if (lightbox.hidden) {
        if (event.key === 'Escape' && !personView.hidden) closePerson();
        return;
      }
      if (event.key === 'Escape') { closeLightbox(); }
      else if (event.key === 'ArrowLeft') { show(currentIndex - 1); }
      else if (event.key === 'ArrowRight') { show(currentIndex + 1); }
    });

    // Swipe on a phone.
    var touchX = null;
    lightbox.addEventListener('touchstart', function (e) {
      touchX = e.changedTouches[0].clientX;
    }, { passive: true });
    lightbox.addEventListener('touchend', function (e) {
      if (touchX === null) return;
      var delta = e.changedTouches[0].clientX - touchX;
      if (Math.abs(delta) > 45) show(currentIndex + (delta < 0 ? 1 : -1));
      touchX = null;
    }, { passive: true });
  }

  /* ===================================================== 5. password gate */
  /* Client-side only: the SHA-256 digest is embedded in the page, so this
     deters casual browsing and nothing more. See the comment in the HTML. */
  function hex(buffer) {
    return Array.prototype.map.call(new Uint8Array(buffer), function (b) {
      return b.toString(16).padStart(2, '0');
    }).join('');
  }

  function initGate() {
    var gate = document.getElementById('gate');
    var form = document.getElementById('gate-form');
    if (!gate || !form) return;

    var app = document.getElementById('app');
    var error = document.getElementById('gate-error');
    var input = document.getElementById('gate-input');

    // Without SubtleCrypto there is nothing honest we can do, so leave the
    // gallery closed rather than silently showing it to anyone.
    if (!window.crypto || !window.crypto.subtle) {
      error.textContent = 'This browser cannot verify the password.';
      error.hidden = false;
      form.querySelector('button').disabled = true;
      return;
    }

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var typed = input.value;
      if (!typed) return;
      window.crypto.subtle.digest('SHA-256', new TextEncoder().encode(typed))
        .then(function (digest) {
          if (hex(digest) === window.__GALLERY_PASSWORD__) {
            gate.remove();
            app.hidden = false;
            document.body.style.overflow = '';
          } else {
            error.hidden = false;
            input.select();
          }
        })
        .catch(function () {
          error.textContent = 'Could not verify the password.';
          error.hidden = false;
        });
    });
  }

  /* ================================================================ boot */
  function boot() {
    initTheme();
    initPeople();
    initLightbox();
    initGate();

    var back = document.getElementById('back');
    if (back) back.addEventListener('click', closePerson);

    // Deep link: gallery.html#Alex opens that person's grid directly.
    var hash = decodeURIComponent((location.hash || '').replace(/^#/, '')).trim();
    if (hash) {
      var index = DATA.findIndex(function (person) {
        return person.name.toLowerCase() === hash.toLowerCase();
      });
      if (index > -1) openPerson(index);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();