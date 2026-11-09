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
