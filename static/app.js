// ozenref — vanilla JS frontend for the recsys lander.
//
// Single user for now (USER_ID hardcoded). Tabs: Feed (personalized),
// All (just the corpus), Similar (top-k near a seed track), Search (text).
// Every action posts to /api/event so the taste vector evolves.

const USER_ID = "alrakhymzhan";
const SESSION_ID = (() => {
  let s = sessionStorage.getItem("session_id");
  if (!s) {
    s = `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem("session_id", s);
  }
  return s;
})();

const $ = (id) => document.getElementById(id);
const grid = $("grid");
const statusEl = $("status");

let state = {
  tab: "feed",
  seed: null,           // track_id of the current seed (Similar tab)
  query: "",
  cards: [],            // last rendered cards
  reactions: new Map(), // track_id → "liked" | "disliked" | "saved"
};

// ---------- API ----------

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

async function logEvent(track_id, action, source, completion_pct = null) {
  try {
    await api("/api/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: USER_ID, track_id, action,
        session_id: SESSION_ID,
        completion_pct, source,
      }),
    });
  } catch (e) {
    console.warn("event log failed:", e);
  }
}

// ---------- Render ----------

function renderEmpty(msg) {
  grid.innerHTML = `<div class="empty">${msg}</div>`;
}

function renderCards(cards, { source = "feed", showSim = false, seedId = null } = {}) {
  state.cards = cards;
  if (!cards.length) {
    renderEmpty("ничего не найдено");
    return;
  }
  grid.innerHTML = "";
  for (const t of cards) {
    const reaction = state.reactions.get(t.track_id) || "";
    const cls = ["card"];
    if (t.track_id === seedId) cls.push("seed");
    if (reaction) cls.push(reaction);
    const card = document.createElement("div");
    card.className = cls.join(" ");
    // Show top-5 tags by sigmoid score, not just best::genre — that's how
    // fusion-genre signal surfaces (a track's best::genre might be
    // "russian pop" while its top fusion tag is "arabian junky drill").
    const tagSpans = (t.top_tags && t.top_tags.length)
      ? t.top_tags.map((tg) =>
          `<span class="tag" title="${tg.group} · ${tg.score}">${escape(tg.tag)}</span>`
        ).join("")
      : [t.best_genre, t.best_mood, t.best_instrument]
          .filter(Boolean)
          .map((x) => `<span class="tag">${escape(x)}</span>`).join("");
    const tags = tagSpans;
    const meta = [
      t.duration_sec ? `${Math.round(t.duration_sec)}s` : null,
      t.bpm ? `${Math.round(t.bpm)} bpm` : null,
      t.key,
    ].filter(Boolean).join(" · ");
    card.innerHTML = `
      <div class="title">${escape(t.title)}</div>
      <div class="tags">${tags}</div>
      <div class="meta">${escape(meta)}${t.sim != null && showSim ? ` · <span class="sim">sim ${t.sim.toFixed(2)}</span>` : ""}</div>
      <div class="actions">
        <button data-act="play"    title="Открыть в Suno">▶ Play</button>
        <button data-act="like"    title="Лайк">👍</button>
        <button data-act="dislike" title="Дизлайк">👎</button>
        <button data-act="save"    title="Сохранить">💾</button>
        <button data-act="similar" title="Похожие">🔁</button>
      </div>
    `;
    card.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", (ev) => onCardAction(ev, t, source))
    );
    grid.appendChild(card);
  }
}

function escape(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

// ---------- Tab handlers ----------

async function loadFeed() {
  setStatus("грузим персональную ленту...");
  const data = await api(`/api/feed?user_id=${USER_ID}&k=20`);
  setStatus(`лента (${data.debug.mode}, ${data.tracks.length} треков)`);
  renderCards(data.tracks, { source: "feed" });
}

async function loadAll() {
  setStatus("грузим корпус...");
  const data = await api(`/api/tracks?limit=200`);
  setStatus(`всего ${data.total} треков`);
  renderCards(data.tracks, { source: "manual" });
}

async function loadSimilar(seedId) {
  if (!seedId) {
    renderEmpty("выбери seed-трек: жми 🔁 на любом треке в других вкладках");
    return;
  }
  setStatus(`похожие на seed ${seedId.slice(0, 8)}...`);
  const data = await api(`/api/similar/${encodeURIComponent(seedId)}?k=10`);
  setStatus(`seed: «${data.seed.title}»  →  10 похожих`);
  renderCards([data.seed, ...data.tracks], {
    source: "similar", showSim: true, seedId,
  });
}

async function loadSearch(q) {
  if (!q) { renderEmpty("введи запрос в шапке"); return; }
  setStatus(`поиск: «${q}»...`);
  const data = await api(`/api/search?q=${encodeURIComponent(q)}&k=10`);
  setStatus(`по запросу «${q}»`);
  renderCards(data.tracks, { source: "search", showSim: true });
}

function setStatus(s) { statusEl.textContent = s; }

async function refresh() {
  try {
    if (state.tab === "feed")    await loadFeed();
    else if (state.tab === "all")     await loadAll();
    else if (state.tab === "similar") await loadSimilar(state.seed);
    else if (state.tab === "search")  await loadSearch(state.query);
  } catch (e) {
    setStatus(`ошибка: ${e.message}`);
  }
}

// ---------- Card actions ----------

async function onCardAction(ev, track, source) {
  ev.stopPropagation();
  const action = ev.currentTarget.dataset.act;
  if (action === "play") {
    if (track.source) window.open(track.source, "_blank");
    await logEvent(track.track_id, "play", source, 0.5);
    return;
  }
  if (action === "similar") {
    state.seed = track.track_id;
    setActiveTab("similar");
    return;
  }
  let logged = action;
  if (action === "like") {
    state.reactions.set(track.track_id, "liked");
  } else if (action === "dislike") {
    state.reactions.set(track.track_id, "disliked");
  } else if (action === "save") {
    state.reactions.set(track.track_id, "saved");
  }
  await logEvent(track.track_id, logged, source);
  // re-render only this card's class
  refresh();
}

// ---------- Tabs / events ----------

function setActiveTab(tab) {
  state.tab = tab;
  document.querySelectorAll("button.tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab)
  );
  refresh();
}

document.querySelectorAll("button.tab").forEach((b) =>
  b.addEventListener("click", () => setActiveTab(b.dataset.tab))
);

$("search-btn").addEventListener("click", () => {
  const q = $("q").value.trim();
  if (!q) return;
  state.query = q;
  setActiveTab("search");
});
$("q").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") $("search-btn").click();
});

$("reload-btn").addEventListener("click", refresh);

// --------- Ingest ---------

$("ingest-btn").addEventListener("click", () => {
  const panel = $("ingest-panel");
  panel.style.display = panel.style.display === "none" ? "" : "none";
  if (panel.style.display !== "none") $("ingest-url").focus();
});
$("ingest-cancel").addEventListener("click", () => {
  $("ingest-panel").style.display = "none";
  $("ingest-url").value = "";
});
$("ingest-url").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") $("ingest-go").click();
  if (ev.key === "Escape") $("ingest-cancel").click();
});
$("ingest-go").addEventListener("click", async () => {
  const url = $("ingest-url").value.trim();
  if (!url) return;
  setStatus(`добавляю трек ${url} ... (~10-30 сек)`);
  $("ingest-go").disabled = true;
  try {
    const r = await fetch("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, keep_audio: false }),
    });
    if (!r.ok) {
      const err = await r.text();
      setStatus(`ошибка: ${err}`);
      return;
    }
    const data = await r.json();
    if (data.status === "already_indexed") {
      setStatus(`этот трек уже в индексе (row ${data.row})`);
    } else {
      const tags = (data.suno_tags || []).map(t => t.tag).join(" / ");
      setStatus(`✓ добавлен: ${data.title}  [${tags}]`);
      // Refresh stats + active tab
      const t = await api(`/api/tracks?limit=1`);
      $("stats").textContent = `${t.total} tracks · user: ${USER_ID}`;
    }
    $("ingest-url").value = "";
    $("ingest-panel").style.display = "none";
    refresh();
  } catch (e) {
    setStatus(`ошибка: ${e.message}`);
  } finally {
    $("ingest-go").disabled = false;
  }
});


$("profile-btn").addEventListener("click", async () => {
  const block = $("profile-block");
  if (block.style.display === "none") {
    const data = await api(`/api/profile/${USER_ID}`);
    $("profile-json").textContent = JSON.stringify(data, null, 2);
    block.style.display = "";
    block.open = true;
  } else {
    block.style.display = "none";
  }
});

// ---------- Init ----------

(async () => {
  try {
    const data = await api(`/api/tracks?limit=1`);
    $("stats").textContent = `${data.total} tracks · user: ${USER_ID}`;
  } catch (e) {
    $("stats").textContent = "(сервер не отвечает)";
  }
  setActiveTab("feed");
})();
