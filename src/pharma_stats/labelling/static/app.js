"use strict";

const state = {
  vocab: null,
  blind: localStorage.getItem("blind_mode") !== "off",
  current: null,        // { serve_token, program }
  prefetched: null,     // { serve_token, program }
  prefetchInFlight: false,
  startedAt: null,
  timerHandle: null,
  gate: 1,               // which gate is currently active: 1, 2, or 3
  form: {
    status: null, kill_reason: null, confidence: null,
    is_adc: null, in_scope: null, scope_reason: null,
  },
  // pass-1 (in-app evidence) vs pass-2 (after checking outbound links) tracking
  firstStatusSelected: null,
  externalLinkClicked: false,
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
  $("gate1Count").textContent = s.gate1_rejected_count;
  $("gate2Count").textContent = s.gate2_rejected_count;
  $("queueCount").textContent = s.remaining_fresh_in_queue;
  $("medianSec").textContent = s.median_seconds_per_label ? s.median_seconds_per_label.toFixed(0) : "–";
  if ($("queueGate1")) $("queueGate1").textContent = s.queue_enter_gate1 ?? "–";
  if ($("queueGate3")) $("queueGate3").textContent = s.queue_enter_gate3 ?? "–";
  if ($("hoursLeft")) {
    $("hoursLeft").textContent = (s.hours_left_to_target != null) ? s.hours_left_to_target.toFixed(1) : "–";
  }

  const patterns = s.gate1_rejection_pattern_counts || [];
  $("gate1PatternBreakdown").textContent = patterns.length
    ? patterns.map(p => {
        const term = p.matched_term ? `:${p.matched_term}` : "";
        return `${p.discovery_strategy}/${p.match_strength}${term}=${p.count}`;
      }).join("  ")
    : "(none yet)";
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
  applyServePlan(data.program);
  startTimer();
  ensurePrefetch();
  refreshSessionStats();
}

// ---------- rendering ----------

function statusPillHtml(status) {
  return `<span class="status-pill status-${status || "UNKNOWN"}">${status || "?"}</span>`;
}

const SCOPE_CATEGORY_ORDER = ["heme", "solid", "non_oncology", "ambiguous"];

function trialScopeCheckHtml(p) {
  const scope = p.trial_scope; // {nct_id: category}, or null (blind validation sample / not classified)
  if (!scope || !Object.keys(scope).length) {
    return `<div style="color:var(--muted)">No MeSH-based scope classification available for this asset's trials.</div>`;
  }
  const byCategory = {};
  for (const [nct, cat] of Object.entries(scope)) {
    (byCategory[cat] = byCategory[cat] || []).push(nct);
  }
  const lines = SCOPE_CATEGORY_ORDER.filter(c => byCategory[c]).map(c =>
    `<div><b>${esc(c)} (${byCategory[c].length}):</b> ${byCategory[c].map(esc).join(", ")}</div>`
  ).join("");

  if (!p.spans_heme_and_solid) {
    return `<div>${lines}</div>`;
  }
  return `
    <div style="border:1px solid var(--warn); border-radius:6px; padding:8px 10px">
      <div style="color:var(--warn); font-weight:600; margin-bottom:4px">
        ⚠ Mixed evidence (MeSH-classified) — this asset has both heme and solid trials. Do not default to heme_only; choose explicitly.
      </div>
      ${lines}
    </div>`;
}

function sponsorClassLine(p) {
  const list = p.sponsors_over_time || [];
  if (!list.length) return `<div class="sponsor-line">Sponsor: none on file</div>`;
  const parts = list.map(s => {
    const cls = s.effective_class || s.class || "UNKNOWN";
    const overridden = s.class_overridden ? ` <span title="sponsor_class_overrides.json">(overridden from ${esc(s.class || "?")})</span>` : "";
    return `${esc(s.sponsor)} <span class="cls">[${esc(cls)}]</span>${overridden}`;
  });
  const cls = p.non_industry_sponsor_hint ? "sponsor-line flagged" : "sponsor-line";
  return `<div class="${cls}">Sponsor: ${parts.join(" &nbsp;·&nbsp; ")}</div>`;
}

function renderProgram(p) {
  const sponsors = (p.sponsors_over_time || []).map(s => {
    const cls = s.effective_class || s.class || "";
    const rawNote = s.class_overridden ? ` <span style="color:var(--muted)">(raw: ${esc(s.class || "?")})</span>` : "";
    return `<tr><td>${esc(s.sponsor)}</td><td>${esc(cls)}${rawNote}</td><td>${esc(s.first_seen || "")}</td><td>${esc(s.last_seen || "")}</td></tr>`;
  }).join("");

  const trials = (p.trials || []).map(t => {
    const cov = (p.trial_coverage || {})[t.nct_id] || "none";
    return `
    <tr>
      <td><a class="outbound-link" href="${t.ctgov_url}" target="_blank">${t.nct_id}</a>
        <span class="badge coverage-${cov}" title="history_coverage">${cov}</span></td>
      <td>${(t.phases || []).join(", ")}</td>
      <td>${statusPillHtml(t.status)}</td>
      <td>${esc(t.start_date || "")}</td>
      <td>${esc(t.primary_completion_date || "")} <span style="color:var(--muted)">${esc(t.primary_completion_type || "")}</span></td>
      <td>${esc(t.completion_date || "")} <span style="color:var(--muted)">${esc(t.completion_type || "")}</span></td>
      <td>${t.enrollment_count ?? ""} <span style="color:var(--muted)">${esc(t.enrollment_type || "")}</span></td>
      <td>${esc(t.why_stopped || "")}</td>
    </tr>`;
  }).join("");

  const timeline = (p.timeline || []).map(e => {
    const isTyped = !!e.event_type;
    const directionBadge = e.direction
      ? `<span class="badge event-direction-${esc(e.direction)}">${esc(e.direction)}</span>` : "";
    const typeBadge = isTyped ? `<span class="badge event-type">${esc(e.event_type)}</span>` : "";
    return `
    <div class="timeline-item">
      <div class="d">${esc(e.date || "?")}</div>
      <div><b>${esc(e.nct_id)}</b> ${typeBadge}${directionBadge} — ${esc(e.label)} ${e.status ? statusPillHtml(e.status) : ""}
        ${e.changed_modules ? `<div style="color:var(--muted);font-size:11px">${esc((e.changed_modules||[]).join(", "))}</div>` : ""}
      </div>
    </div>`;
  }).join("");

  const excludedTrials = (p.excluded_shared_trials || []).map(x => `
    <div class="timeline-item">
      <div class="d"></div>
      <div>⚠ <b>${esc(x.nct_id)}</b> excluded — also independently claimed by
        ${esc((x.shared_with || []).join(", "))}. Likely a genuine combination trial;
        many-to-many trial↔asset linking isn't modelled yet, so this trial contributes to
        neither asset's evidence here rather than being guessed onto one.
        <a class="outbound-link" href="https://clinicaltrials.gov/study/${esc(x.nct_id)}" target="_blank">CT.gov</a>
      </div>
    </div>`).join("");

  const links = [
    ...((p.trials || []).map(t => `<a class="outbound-link" href="${t.ctgov_url}" target="_blank">CT.gov — ${t.nct_id}</a>`)),
    `<a class="outbound-link" href="${p.links.pubmed_search}" target="_blank">PubMed search — ${esc(p.proposed_name)}</a>`,
    `<a class="outbound-link" href="${p.links.web_search_discontinued}" target="_blank">Web search — "${esc(p.proposed_name)}" discontinued</a>`,
  ].join("");

  let revealBadge = "";
  if (p.silence_score !== undefined) {
    revealBadge = `<span class="badge">unblinded — score ${p.silence_score}, band ${p.band}, ${esc((p.archetypes||[]).join("/"))}</span>`;
  }

  const discoveryTerm = p.matched_term ? `: "${esc(p.matched_term)}"` : "";
  const discoveryBadge = `<span class="badge" title="why this candidate exists at all">found via ${esc(p.discovery_strategy || "?")} / ${esc(p.match_strength || "?")}${discoveryTerm}</span>`;

  $("programView").innerHTML = `
    <h1 class="name">${esc(p.proposed_name)}</h1>
    <div class="sub">
      <span class="badge provisional">provisional program = asset only (indication/line not yet normalised)</span>
      <span class="badge">${esc(p.review_status)}</span>
      <span class="badge coverage-${p.history_coverage}" title="history_coverage — always shown, never gated by blind mode">history_coverage: ${esc(p.history_coverage)}</span>
      ${discoveryBadge}
      ${revealBadge}
    </div>
    ${sponsorClassLine(p)}
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

    ${excludedTrials ? `
    <section class="card" style="border-color:var(--warn)">
      <h2 style="color:var(--warn)">Excluded shared trials (${p.excluded_shared_trials.length})</h2>
      ${excludedTrials}
    </section>` : ""}

    <section class="card">
      <h2>Event timeline (typed EvidenceEvents where extracted; falls back to raw amendment history otherwise)</h2>
      ${timeline || "<div style='color:var(--muted)'>no version history indexed for this asset's trials yet</div>"}
    </section>

    <section class="card links">
      <h2>Outbound references</h2>
      ${links}
    </section>
  `;

  $("trialScopeCheck").innerHTML = trialScopeCheckHtml(p);
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

// ---------- judgement panel ----------

function buildChoiceButtons() {
  const isAdcKeys = { yes: "1", no: "2", unsure: "3" };
  $("isAdcChoices").innerHTML = state.vocab.is_adc_values.map((v) =>
    `<button class="choice" data-kind="is_adc" data-value="${v}"><kbd>${isAdcKeys[v] || ""}</kbd>${v}</button>`
  ).join("");

  const inScopeKeys = { yes: "1", no: "2" };
  $("inScopeChoices").innerHTML = state.vocab.in_scope_values.map((v) =>
    `<button class="choice" data-kind="in_scope" data-value="${v}"><kbd>${inScopeKeys[v] || ""}</kbd>${v}</button>`
  ).join("");

  const reasonKeys = "abcd".split("");
  const gate2Reasons = state.vocab.gate2_scope_out_reasons || state.vocab.scope_out_reasons.filter((r) => r !== "not_an_adc");
  $("scopeReasonChoices").innerHTML = gate2Reasons.map((r, i) =>
    `<button class="choice" data-kind="scope_reason" data-value="${r}"><kbd>${reasonKeys[i]}</kbd>${r}</button>`
  ).join("");

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

  // mouse-only — no free keyboard letters left at gate 3 (abcdefgh is kill_reason, 123/shift+123 are status/confidence)
  // (state.vocab.confirmation_evidence_types || []): a stale server predating this field must never
  // throw here and abort the rest of this function — that would silently unbind every OTHER button too
  $("confirmationEvidenceChoices").innerHTML = (state.vocab.confirmation_evidence_types || []).map(t =>
    `<button class="choice" data-kind="confirmation_evidence_type" data-value="${t}">${t}</button>`
  ).join("");

  document.querySelectorAll(".choice").forEach(btn => btn.addEventListener("click", () => {
    const { kind, value } = btn.dataset;
    if (kind === "is_adc") { chooseIsAdc(value); return; }
    if (kind === "in_scope") { chooseInScope(value); return; }
    if (kind === "scope_reason") { chooseScopeReason(value); return; }
    setField(kind, value);
  }));
}

function setField(kind, value) {
  state.form[kind] = value;
  document.querySelectorAll(`.choice[data-kind="${kind}"]`).forEach(b => b.classList.toggle("selected", b.dataset.value === value));
  if (kind === "status") {
    if (state.firstStatusSelected === null) state.firstStatusSelected = value; // pass-1 judgement, captured once
    const isDead = value === "dead_confirmed";
    $("killReasonField").style.display = isDead ? "block" : "none";
    $("deadDatesField").style.display = isDead ? "block" : "none";
    if (!isDead) {
      state.form.kill_reason = null;
      state.form.confirmation_evidence_type = null;
      document.querySelectorAll('.choice[data-kind="kill_reason"]').forEach(b => b.classList.remove("selected"));
      document.querySelectorAll('.choice[data-kind="confirmation_evidence_type"]').forEach(b => b.classList.remove("selected"));
    }
  }
  $("err").textContent = "";
}

// ---------- gate flow ----------
// Gate 1 (is_adc) -> Gate 2 (in_scope, only if is_adc=yes) -> Gate 3 (full
// label, only if in_scope=yes). Gate 1/2 rejections save immediately —
// there is nothing further to fill in, and speed is the point.

function showGate(n) {
  state.gate = n;
  $("gate1").style.display = (n === 1 || state.gatesUnlocked) ? "block" : "none";
  $("gate2").style.display = n >= 2 ? "block" : "none";
  $("gate3").style.display = n >= 3 ? "block" : "none";
  $("gate3Actions").style.display = n >= 3 ? "flex" : "none";
  // Auto-derived skip of earlier gates: hide them unless the reviewer
  // explicitly unlocked override.
  if (!state.gatesUnlocked && n === 3) {
    $("gate1").style.display = "none";
    $("gate2").style.display = "none";
  }
  if (!state.gatesUnlocked && n === 2) {
    $("gate1").style.display = "none";
  }
}

// Cheap MeSH-derived pre-fill, shown as a hint the reviewer can override —
// never auto-submitted. scope_category="heme_only" only ever reaches this
// screen for an asset the auto-exclusion script hasn't processed yet (or
// one in the blind validation sample, where scope_category is withheld
// entirely — see app.py's _program_public).
function scopeHintReason(p) {
  if (p.scope_category === "heme_only") return "heme_only";
  if (p.non_oncology_hint) return "non_oncology";
  if (p.non_industry_sponsor_hint) return "non_industry";
  return null;
}

function chooseIsAdc(value) {
  setField("is_adc", value);
  if (value === "yes") {
    const ctx = state.current && state.current.program && state.current.program.triage_context;
    if (ctx && ctx.in_scope === "yes" && !state.gatesUnlocked) {
      setField("in_scope", "yes");
      showGate(3);
      return;
    }
    showGate(2);
    const reason = scopeHintReason(state.current.program);
    if (reason) {
      setField("in_scope", "no");
      setField("scope_reason", reason);
      $("scopeReasonField").style.display = "block";
    }
  } else {
    if (value === "no") {
      setField("in_scope", "no");
      setField("scope_reason", "not_an_adc");
    }
    submitLabel(1);
  }
}

function chooseInScope(value) {
  setField("in_scope", value);
  if (value === "yes") {
    showGate(3);
  } else {
    $("scopeReasonField").style.display = "block";
  }
}

function chooseScopeReason(value) {
  setField("scope_reason", value);
  submitLabel(2);
}

function resetForm() {
  state.form = {
    status: null, kill_reason: null, confidence: null,
    is_adc: null, in_scope: null, scope_reason: null,
    confirmation_evidence_type: null,
  };
  state.firstStatusSelected = null;
  state.externalLinkClicked = false;
  state.gatesUnlocked = false;
  showGate(1);
  document.querySelectorAll(".choice").forEach(b => b.classList.remove("selected"));
  $("killReasonField").style.display = "none";
  $("deadDatesField").style.display = "none";
  $("scopeReasonField").style.display = "none";
  $("evidenceNote").value = "";
  $("labelEvidenceDate").value = "";
  $("publicConfirmationDate").value = "";
  $("neverConfirmed").checked = false;
  $("thirdPartyFirstNotedDate").value = "";
  $("thirdPartySource").value = "";
  $("err").textContent = "";
}

function overrideTriageGates() {
  state.gatesUnlocked = true;
  showGate(1);
}

function renderTriageBanners(p) {
  const reopenEl = $("reopenBanner");
  const triageEl = $("triageBanner");
  if (reopenEl) {
    if (p.reopened) {
      reopenEl.style.display = "block";
      reopenEl.innerHTML = `<section class="card reopen-banner"><h2>Re-opened for re-decision</h2>
        <div>Previous gold line is unchanged. This is a new review — Gates 1–3 from scratch, no pre-fill.</div></section>`;
    } else {
      reopenEl.style.display = "none";
      reopenEl.innerHTML = "";
    }
  }
  if (!triageEl) return;
  const ctx = p.triage_context;
  const start = p.start_gate || 1;
  if (!ctx || p.reopened || start === 1) {
    triageEl.style.display = "none";
    triageEl.innerHTML = "";
    return;
  }
  const quote = ctx.quote
    ? `<blockquote style="margin:8px 0; color:var(--text)">${esc(ctx.quote)}</blockquote>`
    : "";
  const source = ctx.source_url
    ? `<div class="sub"><a href="${esc(ctx.source_url)}" target="_blank">${esc(ctx.source_url)}</a></div>`
    : "";
  const rule = ctx.rule || ctx.evidence || "";
  triageEl.style.display = "block";
  triageEl.innerHTML = `<section class="card triage-banner">
    <h2>Auto-derived — override if wrong</h2>
    <div><b>is_adc=${esc(ctx.is_adc || "?")}</b> · <b>in_scope=${esc(ctx.in_scope || "?")}</b>
      ${ctx.scope_reason ? ` (${esc(ctx.scope_reason)})` : ""}
      · layer ${esc(ctx.layer)} · ${esc(rule)}</div>
    ${quote}${source}
    <div id="triageOverride"><button class="secondary" type="button" id="overrideTriageBtn">Re-answer Gates 1–2</button></div>
  </section>`;
  const btn = $("overrideTriageBtn");
  if (btn) btn.addEventListener("click", overrideTriageGates);
}

function applyServePlan(p) {
  renderTriageBanners(p);
  const start = p.start_gate || 1;
  const ctx = p.triage_context;
  if (p.reopened || start === 1) {
    showGate(1);
    return;
  }
  if (ctx && ctx.is_adc) setField("is_adc", ctx.is_adc);
  if (start >= 2 && ctx && ctx.in_scope) setField("in_scope", ctx.in_scope);
  if (start >= 3) {
    setField("is_adc", "yes");
    setField("in_scope", "yes");
  }
  showGate(start);
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

// ---------- save / skip ----------

async function submitLabel(gateReached) {
  if (!state.current) return;
  const payload = {
    serve_token: state.current.serve_token,
    action: "label",
    gate_reached: gateReached,
    is_adc: state.form.is_adc,
    in_scope: state.form.in_scope,
    scope_reason: state.form.scope_reason,
    status: state.form.status,
    kill_reason: state.form.kill_reason,
    confidence: state.form.confidence,
    evidence_note: $("evidenceNote").value,
    label_evidence_date: $("labelEvidenceDate").value || null,
    public_confirmation_date: $("publicConfirmationDate").value || null,
    confirmation_evidence_type: state.form.confirmation_evidence_type,
    never_publicly_confirmed: $("neverConfirmed").checked,
    third_party_first_noted_date: $("thirdPartyFirstNotedDate").value || null,
    third_party_source: $("thirdPartySource").value || null,
    blind: state.blind,
    seconds_spent: secondsSpent(),
    // pass-2 revision: only meaningful at gate 3, and only if they actually
    // left the app to check something AND the status they ended up saving
    // differs from the one they first picked from in-app evidence alone
    status_revised_after_external_search:
      gateReached === 3 &&
      state.externalLinkClicked &&
      state.firstStatusSelected !== null &&
      state.firstStatusSelected !== state.form.status,
  };
  await sendAndAdvance(payload, gateReached === 3);
}

async function submitSkip() {
  if (!state.current) return;
  await sendAndAdvance({
    serve_token: state.current.serve_token,
    action: "skip",
    blind: state.blind,
    seconds_spent: secondsSpent(),
  }, false);
}

async function sendAndAdvance(payload, mayReveal) {
  const r = await fetch("/api/labels", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });

  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    $("err").textContent = body.detail || `save failed (${r.status})`;
    return;
  }

  const result = await r.json();
  if (mayReveal && result.reveal) {
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
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && state.gate === 3) { e.preventDefault(); submitLabel(3); }
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }

  if (e.altKey && e.key.toLowerCase() === "b") { e.preventDefault(); setBlind(!state.blind); return; }
  if (e.key === "?") { $("help").classList.add("show"); return; }
  if (e.key.toLowerCase() === "n") { submitSkip(); return; }

  if (state.gate === 1) {
    const idx = { "1": "yes", "2": "no", "3": "unsure" }[e.key];
    if (idx && state.vocab && state.vocab.is_adc_values.includes(idx)) { chooseIsAdc(idx); }
    return;
  }

  if (state.gate === 2) {
    if (state.form.in_scope !== "no") {
      const idx = { "1": "yes", "2": "no" }[e.key];
      if (idx && state.vocab && state.vocab.in_scope_values.includes(idx)) { chooseInScope(idx); }
      return;
    }
    const reasonIdx = "abcd".indexOf(e.key.toLowerCase());
    if (reasonIdx >= 0 && state.vocab && state.vocab.scope_out_reasons[reasonIdx]) {
      chooseScopeReason(state.vocab.scope_out_reasons[reasonIdx]);
    }
    return;
  }

  // gate === 3
  if (e.key === "Enter") { submitLabel(3); return; }
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

$("saveBtn").addEventListener("click", () => submitLabel(3));
$("skipBtn").addEventListener("click", () => submitSkip());
$("blindCheckbox").addEventListener("change", (e) => setBlind(e.target.checked));
$("helpBtn").addEventListener("click", () => $("help").classList.add("show"));
$("helpClose").addEventListener("click", () => $("help").classList.remove("show"));

// Delegated: outbound links (PubMed / web search / CT.gov) are rendered
// fresh into #programView on every program, so a single listener here
// catches all of them without re-binding per render. This is pass-2 —
// checking evidence outside the app — for status_revised_after_external_search.
$("programView").addEventListener("click", (e) => {
  if (e.target.closest("a.outbound-link")) state.externalLinkClicked = true;
});

// ---------- boot ----------

(async function boot() {
  setBlind(state.blind);
  await loadVocab();
  await refreshSessionStats();
  await showNext();
})();
