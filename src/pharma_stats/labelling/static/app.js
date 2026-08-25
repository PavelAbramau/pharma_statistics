"use strict";

const state = {
  vocab: null,
  blind: localStorage.getItem("blind_mode") !== "off",
  current: null,        // { serve_token, program }
  prefetched: null,     // { serve_token, program }
  prefetchInFlight: false,
  startedAt: null,
  timerHandle: null,
  form: { status: null, kill_reason: null, confidence: null },
};

const $ = (id) => document.getElementById(id);

function fmtSeconds(s) { return s.toFixed(1); }

// ---------- data loading ----------

async function loadVocab() {
  const r = await fetch("/api/vocab");
  state.vocab = await r.json();
  buildChoiceButtons();
}

async function refreshSessionStats() {
  const r = await fetch("/api/session");
  const s = await r.json();
  $("labelledCount").textContent = s.labelled_count;
  $("totalCount").textContent = s.total_programs;
  $("queueCount").textContent = s.remaining_fresh_in_queue;
  $("medianSec").textContent = s.median_seconds_per_label ? s.median_seconds_per_label.toFixed(0) : "–";
}

async function fetchNext() {
  const r = await fetch(`/api/next?blind=${state.blind}`);
  return r.json();
}

async function ensurePrefetch() {
  if (state.prefetched || state.prefetchInFlight) return;
  state.prefetchInFlight = true;
  try {
    const data = await fetchNext();
    state.prefetched = data;
  } finally {
    state.prefetchInFlight = false;
  }
}

async function showNext() {
  let data;
  if (state.prefetched) {
    data = state.prefetched;
    state.prefetched = null;
  } else {
    data = await fetchNext();
  }
  if (data.done) {
    document.getElementById("layout").style.display = "none";
    document.getElementById("topbar").style.display = "none";
    document.getElementById("done").style.display = "block";
    return;
  }
  state.current = data;
  resetForm();
  renderProgram(data.program);
  startTimer();
  ensurePrefetch();
  refreshSessionStats();
}

// ---------- rendering ----------

function statusPillHtml(status) {
  return `<span class="status-pill status-${status || "UNKNOWN"}">${status || "?"}</span>`;
}

function renderProgram(p) {
  const sponsors = (p.sponsors_over_time || []).map(s =>
    `<tr><td>${esc(s.sponsor)}</td><td>${esc(s.class || "")}</td><td>${esc(s.first_seen || "")}</td><td>${esc(s.last_seen || "")}</td></tr>`
  ).join("");

  const trials = (p.trials || []).map(t => `
    <tr>
      <td><a href="${t.ctgov_url}" target="_blank">${t.nct_id}</a></td>
      <td>${(t.phases || []).join(", ")}</td>
      <td>${statusPillHtml(t.status)}</td>
      <td>${esc(t.start_date || "")}</td>
      <td>${esc(t.primary_completion_date || "")} <span style="color:var(--muted)">${esc(t.primary_completion_type || "")}</span></td>
      <td>${esc(t.completion_date || "")} <span style="color:var(--muted)">${esc(t.completion_type || "")}</span></td>
      <td>${t.enrollment_count ?? ""} <span style="color:var(--muted)">${esc(t.enrollment_type || "")}</span></td>
      <td>${esc(t.why_stopped || "")}</td>
    </tr>`).join("");

  const timeline = (p.timeline || []).map(e => `
    <div class="timeline-item">
      <div class="d">${esc(e.date || "?")}</div>
      <div><b>${esc(e.nct_id)}</b> — ${esc(e.label)} ${e.status ? statusPillHtml(e.status) : ""}
        ${e.changed_modules ? `<div style="color:var(--muted);font-size:11px">${esc((e.changed_modules||[]).join(", "))}</div>` : ""}
      </div>
    </div>`).join("");

  const links = [
    ...((p.trials || []).map(t => `<a href="${t.ctgov_url}" target="_blank">CT.gov — ${t.nct_id}</a>`)),
    `<a href="${p.links.pubmed_search}" target="_blank">PubMed search — ${esc(p.proposed_name)}</a>`,
    `<a href="${p.links.web_search_discontinued}" target="_blank">Web search — "${esc(p.proposed_name)}" discontinued</a>`,
  ].join("");

  let revealBadge = "";
  if (p.silence_score !== undefined) {
    revealBadge = `<span class="badge">unblinded — score ${p.silence_score}, band ${p.band}, ${esc((p.archetypes||[]).join("/"))}</span>`;
  }

  $("programView").innerHTML = `
    <h1 class="name">${esc(p.proposed_name)}</h1>
    <div class="sub">
      <span class="badge provisional">provisional program = asset only (indication/line not yet normalised)</span>
      <span class="badge">${esc(p.review_status)}</span>
      ${revealBadge}
    </div>
    ${p.synonyms.length ? `<div class="sub">Synonyms: ${esc(p.synonyms.join(", "))}</div>` : ""}

    <section class="card">
      <h2>Sponsor / ownership history</h2>
      <table><tr><th>Sponsor</th><th>Class</th><th>First seen</th><th>Last seen</th></tr>${sponsors || "<tr><td colspan=4 style='color:var(--muted)'>none on file</td></tr>"}</table>
    </section>

    <section class="card">
      <h2>Trials (${p.trial_count})</h2>
      <table>
        <tr><th>NCT</th><th>Phase</th><th>Status</th><th>Start</th><th>Primary compl.</th><th>Completion</th><th>Enrollment</th><th>Why stopped</th></tr>
        ${trials || "<tr><td colspan=8 style='color:var(--muted)'>no snapshot on file for this candidate's trials</td></tr>"}
      </table>
    </section>

    <section class="card">
      <h2>Event timeline (raw amendment history — untyped; EvidenceEvent extraction not built yet)</h2>
      ${timeline || "<div style='color:var(--muted)'>no version history indexed for this asset's trials yet</div>"}
    </section>

    <section class="card links">
      <h2>Outbound references</h2>
      ${links}
    </section>
  `;
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

// ---------- judgement panel ----------

function buildChoiceButtons() {
  const statusKeys = "123456".split("");
  $("statusChoices").innerHTML = state.vocab.statuses.map((s, i) =>
    `<button class="choice" data-kind="status" data-value="${s}"><kbd>${statusKeys[i]}</kbd>${s}</button>`
  ).join("");

  const killKeys = "abcdefgh".split("");
  $("killChoices").innerHTML = state.vocab.kill_reasons.map((k, i) =>
    `<button class="choice" data-kind="kill_reason" data-value="${k}"><kbd>${killKeys[i]}</kbd>${k}</button>`
  ).join("");

  const confLabels = { high: "Shift+1", medium: "Shift+2", low: "Shift+3" };
  $("confidenceChoices").innerHTML = state.vocab.confidence_levels.map(c =>
    `<button class="choice" data-kind="confidence" data-value="${c}"><kbd>${confLabels[c]}</kbd>${c}</button>`
  ).join("");

  document.querySelectorAll(".choice").forEach(btn => btn.addEventListener("click", () => setField(btn.dataset.kind, btn.dataset.value)));
}

function setField(kind, value) {
  state.form[kind] = value;
  document.querySelectorAll(`.choice[data-kind="${kind}"]`).forEach(b => b.classList.toggle("selected", b.dataset.value === value));
  if (kind === "status") {
    const isDead = value === "dead_confirmed";
    $("killReasonField").style.display = isDead ? "block" : "none";
    $("deadDatesField").style.display = isDead ? "block" : "none";
    if (!isDead) { state.form.kill_reason = null; document.querySelectorAll('.choice[data-kind="kill_reason"]').forEach(b => b.classList.remove("selected")); }
  }
  $("err").textContent = "";
}

function resetForm() {
  state.form = { status: null, kill_reason: null, confidence: null };
  document.querySelectorAll(".choice").forEach(b => b.classList.remove("selected"));
  $("killReasonField").style.display = "none";
  $("deadDatesField").style.display = "none";
  $("evidenceNote").value = "";
  $("labelEvidenceDate").value = "";
  $("publicConfirmationDate").value = "";
  $("neverConfirmed").checked = false;
  $("err").textContent = "";
}

// ---------- timer ----------

function startTimer() {
  state.startedAt = performance.now();
  if (state.timerHandle) clearInterval(state.timerHandle);
  state.timerHandle = setInterval(() => {
    $("timer").textContent = fmtSeconds((performance.now() - state.startedAt) / 1000) + "s";
  }, 100);
}

function secondsSpent() {
  return (performance.now() - state.startedAt) / 1000;
}

// ---------- save / skip / flag ----------

async function submit(action) {
  if (!state.current) return;
  const payload = {
    serve_token: state.current.serve_token,
    action,
    status: state.form.status,
    kill_reason: state.form.kill_reason,
    confidence: state.form.confidence,
    evidence_note: $("evidenceNote").value,
    label_evidence_date: $("labelEvidenceDate").value || null,
    public_confirmation_date: $("publicConfirmationDate").value || null,
    never_publicly_confirmed: $("neverConfirmed").checked,
    blind: state.blind,
    seconds_spent: secondsSpent(),
  };

  const r = await fetch("/api/labels", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });

  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    $("err").textContent = body.detail || `save failed (${r.status})`;
    return;
  }

  const result = await r.json();
  if (action === "label" && result.reveal) {
    showReveal(result.reveal);
  } else {
    showNext();
  }
}

function showReveal(reveal) {
  $("revealScore").textContent = reveal.silence_score;
  $("revealBand").textContent = reveal.band;
  $("revealBar").style.width = reveal.silence_score + "%";
  $("revealArchetypes").textContent = (reveal.archetypes || []).join(", ");
  $("revealBreakdown").innerHTML = Object.entries(reveal.score_breakdown || {})
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("");
  $("reveal").classList.add("show");
  const dismiss = () => { $("reveal").classList.remove("show"); showNext(); };
  $("revealClose").onclick = dismiss;
  state._dismissReveal = dismiss;
}

// ---------- blind toggle ----------

function setBlind(on) {
  state.blind = on;
  localStorage.setItem("blind_mode", on ? "on" : "off");
  $("blindCheckbox").checked = on;
  $("blindToggle").classList.toggle("on", on);
  $("blindToggle").classList.toggle("off", !on);
  state.prefetched = null; // stale w.r.t. new blind setting
  ensurePrefetch();
}

// ---------- keyboard ----------

document.addEventListener("keydown", (e) => {
  const inField = document.activeElement && ["TEXTAREA", "INPUT"].includes(document.activeElement.tagName);

  if ($("reveal").classList.contains("show")) {
    if (state._dismissReveal) state._dismissReveal();
    return;
  }
  if ($("help").classList.contains("show")) {
    if (e.key === "Escape") $("help").classList.remove("show");
    return;
  }

  if (inField) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit("label"); }
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }

  if (e.altKey && e.key.toLowerCase() === "b") { e.preventDefault(); setBlind(!state.blind); return; }
  if (e.key === "?") { $("help").classList.add("show"); return; }
  if (e.key === "Enter") { submit("label"); return; }
  if (e.key.toLowerCase() === "n") { submit("skip"); return; }
  if (e.key.toLowerCase() === "x") { submit("flag_invalid"); return; }

  if (e.shiftKey && ["1", "2", "3"].includes(e.key)) {
    const conf = { "1": "high", "2": "medium", "3": "low" }[e.key];
    setField("confidence", conf);
    return;
  }
  if ("123456".includes(e.key)) {
    const idx = parseInt(e.key, 10) - 1;
    if (state.vocab && state.vocab.statuses[idx]) setField("status", state.vocab.statuses[idx]);
    return;
  }
  if ("abcdefgh".includes(e.key.toLowerCase()) && state.form.status === "dead_confirmed") {
    const idx = "abcdefgh".indexOf(e.key.toLowerCase());
    if (state.vocab && state.vocab.kill_reasons[idx]) setField("kill_reason", state.vocab.kill_reasons[idx]);
    return;
  }
});

$("saveBtn").addEventListener("click", () => submit("label"));
$("skipBtn").addEventListener("click", () => submit("skip"));
$("invalidBtn").addEventListener("click", () => submit("flag_invalid"));
$("blindCheckbox").addEventListener("change", (e) => setBlind(e.target.checked));
$("helpBtn").addEventListener("click", () => $("help").classList.add("show"));
$("helpClose").addEventListener("click", () => $("help").classList.remove("show"));

// ---------- boot ----------

(async function boot() {
  setBlind(state.blind);
  await loadVocab();
  await refreshSessionStats();
  await showNext();
})();
