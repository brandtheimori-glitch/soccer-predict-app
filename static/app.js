document.querySelectorAll(".why-toggle").forEach((btn) => {
  btn.addEventListener("click", () => {
    const panel = btn.nextElementSibling;
    panel.classList.toggle("hidden");
    btn.textContent = panel.classList.contains("hidden") ? "why? ▾" : "why? ▴";
  });
});

const recordToggle = document.getElementById("toggle-record");
if (recordToggle) {
  recordToggle.addEventListener("click", () => {
    const table = document.getElementById("record-table");
    table.classList.toggle("hidden");
    recordToggle.textContent = table.classList.contains("hidden") ? "show log" : "hide log";
  });
}

const valueBetToggle = document.getElementById("toggle-value-bets");
if (valueBetToggle) {
  valueBetToggle.addEventListener("click", () => {
    const table = document.getElementById("value-bet-table");
    table.classList.toggle("hidden");
    valueBetToggle.textContent = table.classList.contains("hidden") ? "show bets" : "hide bets";
  });
}

// --- Install as an app -----------------------------------------------------
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}

let deferredInstall = null;
const installBtn = document.getElementById("install-app");
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredInstall = e;
  if (installBtn) installBtn.hidden = false;
});
if (installBtn) {
  installBtn.addEventListener("click", async () => {
    if (!deferredInstall) return;
    deferredInstall.prompt();
    await deferredInstall.userChoice;
    deferredInstall = null;
    installBtn.hidden = true;
  });
}
window.addEventListener("appinstalled", () => {
  if (installBtn) installBtn.hidden = true;
});
