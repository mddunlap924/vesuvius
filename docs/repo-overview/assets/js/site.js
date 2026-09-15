/* ==========================================================================
   Vesuvius — Repo Overview documentation portal
   Progressive enhancement only: the site is fully readable without this file.
   ========================================================================== */

(function () {
  "use strict";

  /* --- Sidebar (mobile) ---------------------------------------------------- */

  function initSidebar() {
    var toggle = document.querySelector(".topbar__toggle");
    if (!toggle) return;

    var scrim = document.querySelector(".sidebar__scrim");
    if (!scrim) {
      scrim = document.createElement("div");
      scrim.className = "sidebar__scrim";
      document.body.appendChild(scrim);
    }

    function close() {
      document.body.classList.remove("nav-open");
      toggle.setAttribute("aria-expanded", "false");
    }

    function open() {
      document.body.classList.add("nav-open");
      toggle.setAttribute("aria-expanded", "true");
    }

    toggle.addEventListener("click", function () {
      if (document.body.classList.contains("nav-open")) close();
      else open();
    });

    scrim.addEventListener("click", close);

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") close();
    });

    // Navigating away via a sidebar link should dismiss the drawer.
    document.querySelectorAll(".sidebar a").forEach(function (link) {
      link.addEventListener("click", close);
    });
  }

  /* --- Active nav link ----------------------------------------------------- */

  function initActiveNav() {
    var here = new URL(window.location.href);
    // Treat ".../section/" and ".../section/index.html" as the same page.
    var herePath = here.pathname.replace(/index\.html$/, "");

    document.querySelectorAll(".nav__link").forEach(function (link) {
      var target;
      try {
        target = new URL(link.getAttribute("href"), here);
      } catch (err) {
        return;
      }
      var targetPath = target.pathname.replace(/index\.html$/, "");
      if (targetPath === herePath) {
        link.classList.add("is-active");
        link.setAttribute("aria-current", "page");
      }
    });
  }

  /* --- In-page table of contents ------------------------------------------- */

  function initToc() {
    var host = document.querySelector("[data-toc]");
    if (!host) return;

    var headings = Array.prototype.slice.call(
      document.querySelectorAll(".content h2[id]")
    );
    if (headings.length < 3) {
      host.remove();
      return;
    }

    var title = document.createElement("div");
    title.className = "toc__title";
    title.textContent = "On this page";

    var list = document.createElement("ol");
    headings.forEach(function (heading) {
      var item = document.createElement("li");
      var anchor = document.createElement("a");
      anchor.href = "#" + heading.id;
      // Strip the auto-numbering / punctuation we author into the heading text.
      anchor.textContent = heading.textContent.replace(/^\s*\d+\.\s*/, "");
      item.appendChild(anchor);
      list.appendChild(item);
    });

    host.appendChild(title);
    host.appendChild(list);
    host.hidden = false;
  }

  /* --- Heading anchors ----------------------------------------------------- */

  function initAnchors() {
    document.querySelectorAll(".content h2[id], .content h3[id]").forEach(function (heading) {
      var anchor = document.createElement("a");
      anchor.className = "anchor";
      anchor.href = "#" + heading.id;
      anchor.setAttribute("aria-hidden", "true");
      anchor.textContent = "#";
      anchor.style.cssText =
        "margin-left:.4rem;font-size:.75em;color:var(--fg-subtle);opacity:0;text-decoration:none";
      heading.appendChild(anchor);

      heading.addEventListener("mouseenter", function () {
        anchor.style.opacity = "1";
      });
      heading.addEventListener("mouseleave", function () {
        anchor.style.opacity = "0";
      });
    });
  }

  /* --- Copy buttons on code blocks ----------------------------------------- */

  function initCopyButtons() {
    document.querySelectorAll("pre").forEach(function (block) {
      var code = block.querySelector("code");
      if (!code) return;

      var button = document.createElement("button");
      button.type = "button";
      button.className = "copy-btn";
      button.textContent = "Copy";

      button.addEventListener("click", function () {
        var text = code.textContent;
        var done = function () {
          button.textContent = "Copied";
          window.setTimeout(function () {
            button.textContent = "Copy";
          }, 1200);
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            button.textContent = "Press Ctrl+C";
          });
        } else {
          var area = document.createElement("textarea");
          area.value = text;
          document.body.appendChild(area);
          area.select();
          try {
            document.execCommand("copy");
            done();
          } catch (err) {
            button.textContent = "Press Ctrl+C";
          }
          document.body.removeChild(area);
        }
      });

      block.appendChild(button);
    });
  }

  /* --- Theme toggle -------------------------------------------------------- */

  function initTheme() {
    var button = document.querySelector(".theme-toggle");
    if (!button) return;

    function label() {
      var isDark = document.documentElement.getAttribute("data-theme") === "dark";
      button.textContent = isDark ? "Light" : "Dark";
      button.setAttribute(
        "aria-label",
        isDark ? "Switch to light theme" : "Switch to dark theme"
      );
    }

    button.addEventListener("click", function () {
      var isDark = document.documentElement.getAttribute("data-theme") === "dark";
      var next = isDark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        window.localStorage.setItem("vesuvius-docs-theme", next);
      } catch (err) {
        /* storage unavailable — theme simply won't persist */
      }
      label();
    });

    label();
  }

  /* --- Boot ---------------------------------------------------------------- */

  function boot() {
    initSidebar();
    initActiveNav();
    initToc();
    initAnchors();
    initCopyButtons();
    initTheme();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
