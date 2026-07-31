const themeToggle = document.querySelector("#theme-toggle");
const storedTheme = localStorage.getItem("evidence-theme");
const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const icon = themeToggle?.querySelector("i");
  if (icon) icon.setAttribute("data-lucide", theme === "dark" ? "sun" : "moon");
  if (window.lucide) window.lucide.createIcons();
}

applyTheme(storedTheme || (preferredDark ? "dark" : "light"));

themeToggle?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("evidence-theme", next);
  applyTheme(next);
});

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    const filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelectorAll(".media-card").forEach((card) => {
      card.hidden = filter !== "all" && card.dataset.outcome !== filter;
    });
  });
});

function rateCell(successes, episodes, model) {
  const rate = episodes ? (successes / episodes) * 100 : 0;
  return `
    <div class="rate-row">
      <div class="mini-track" aria-hidden="true"><span class="${model}-bar" style="width:${rate}%"></span></div>
      <span class="rate-number">${successes}/${episodes}</span>
    </div>`;
}

async function renderRoboCasaTable() {
  const body = document.querySelector("#task-table-body");
  if (!body) return;

  try {
    const response = await fetch("data/robocasa-official-match.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    const gr00t = new Map(result.models.gr00t.tasks.map((item) => [item.task, item]));
    const pi05 = new Map(result.models.pi05.tasks.map((item) => [item.task, item]));

    body.innerHTML = result.protocol.tasks.map((task) => {
      const g = gr00t.get(task);
      const p = pi05.get(task);
      const delta = ((g.success_rate - p.success_rate) * 100).toFixed(0);
      const deltaClass = Number(delta) >= 0 ? "delta-positive" : "delta-negative";
      return `
        <tr>
          <td class="task-name">${task}</td>
          <td class="rate-cell">${rateCell(g.successes, g.episodes, "gr00t")}</td>
          <td class="rate-cell">${rateCell(p.successes, p.episodes, "pi05")}</td>
          <td class="${deltaClass}">${Number(delta) > 0 ? "+" : ""}${delta} pt</td>
        </tr>`;
    }).join("");
  } catch (error) {
    body.innerHTML = `<tr><td colspan="4" class="loading-row">结果 JSON 读取失败：${error.message}</td></tr>`;
  }
}

renderRoboCasaTable();
window.addEventListener("DOMContentLoaded", () => window.lucide?.createIcons());
