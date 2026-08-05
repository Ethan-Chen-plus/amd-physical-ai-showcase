const themeToggle = document.querySelector("#theme-toggle");
const langToggle = document.querySelector("#lang-toggle");
const langLabel = document.querySelector("#lang-label");
const storedTheme = localStorage.getItem("evidence-theme");
const storedLanguage = localStorage.getItem("evidence-language") || "en";
const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const icon = themeToggle?.querySelector("i");
  if (icon) icon.setAttribute("data-lucide", theme === "dark" ? "sun" : "moon");
  window.lucide?.createIcons();
}

function applyLanguage(language) {
  const lang = language === "zh" ? "zh" : "en";
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
  document.documentElement.dataset.language = lang;
  document.querySelectorAll("[data-zh][data-en]").forEach((element) => {
    element.textContent = element.dataset[lang];
  });
  document.querySelectorAll("[data-zh-title][data-en-title]").forEach((element) => {
    element.title = element.dataset[lang === "en" ? "enTitle" : "zhTitle"];
  });
  if (langLabel) langLabel.textContent = lang === "en" ? "中" : "EN";
  localStorage.setItem("evidence-language", lang);
  window.lucide?.createIcons();
}

applyTheme(storedTheme || (preferredDark ? "dark" : "light"));
applyLanguage(storedLanguage);

themeToggle?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("evidence-theme", next);
  applyTheme(next);
});

langToggle?.addEventListener("click", () => {
  applyLanguage(document.documentElement.dataset.language === "en" ? "zh" : "en");
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
    const message = document.documentElement.dataset.language === "en" ? "Result JSON could not be loaded" : "结果 JSON 读取失败";
    body.innerHTML = `<tr><td colspan="4" class="loading-row">${message}: ${error.message}</td></tr>`;
  }
}

renderRoboCasaTable();
window.addEventListener("DOMContentLoaded", () => window.lucide?.createIcons());
