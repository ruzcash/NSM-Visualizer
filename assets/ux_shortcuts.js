(function () {
  "use strict";

  function initUxShortcuts() {
    if (window.__nsmUxShortcutsInit) {
      return;
    }
    window.__nsmUxShortcutsInit = true;

    document.addEventListener(
      "keydown",
      function (event) {
        var isEnter =
          event.key === "Enter" ||
          event.code === "Enter" ||
          event.code === "NumpadEnter";
        if (!isEnter || !event.ctrlKey) {
          return;
        }
        var active = document.activeElement;
        var tag = active && active.tagName ? active.tagName.toLowerCase() : "";
        if (tag === "textarea") {
          return;
        }
        var runBtn = document.getElementById("run_calc");
        if (!runBtn || runBtn.disabled) {
          return;
        }
        event.preventDefault();
        runBtn.click();
      },
      true
    );

    document.addEventListener(
      "click",
      function (event) {
        var target = event.target;
        var runBtn = target && target.closest ? target.closest("#run_calc") : null;
        if (!runBtn) {
          return;
        }
        // Auto-scroll only when user is near the top area with controls.
        if (window.scrollY > 260) {
          return;
        }
        var chartAnchor =
          document.getElementById("chart_title") ||
          document.getElementById("chart_wrap") ||
          document.getElementById("supply_chart");
        if (!chartAnchor || !chartAnchor.scrollIntoView) {
          return;
        }
        window.setTimeout(function () {
          chartAnchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 40);
      },
      true
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initUxShortcuts);
  } else {
    initUxShortcuts();
  }
})();
