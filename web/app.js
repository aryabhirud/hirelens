// HireLens frontend. No build step: plain ES modules talking to the FastAPI backend.
// Everything the agents produce is inserted with textContent (never innerHTML),
// because it is model output.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function fill(el, ...children) {
  el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "style") el.style.cssText = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const AGENT_NAMES = {
  orchestrator: "Orchestrator",
  resume_parser: "Resume Parser",
  job_analyst: "Job Analyst",
  matcher: "Matcher",
  critic: "Critic",
  career_coach: "Career Coach",
  finalize: "report",
};
const SECTION_NAMES = {
  skills: "Skills", experience: "Experience", projects: "Projects", education: "Education", keywords: "Keywords",
};
const ACTIVITY = {
  orchestrator: "Planning the next step",
  resume_parser: "Reading your resume",
  job_analyst: "Working out what the role needs",
  matcher: "Scoring your fit, section by section",
  critic: "Checking each score against your resume",
  career_coach: "Writing your suggestions",
};
const QUIET_EVENTS = new Set(["tool_result", "handoff"]);
const MAX_PDF_BYTES = 4 * 1024 * 1024; // the hosting platform accepts uploads up to about 4.5 MB

const state = {
  file: null,
  runId: null,
  result: null,
  events: [],
  seen: new Set(),
  active: null,
  errored: new Set(),
};

// ---------------------------------------------------------------- views
function showView(name) {
  for (const view of ["intake", "run", "report"]) {
    $(`#view-${view}`).hidden = view !== name;
  }
  $("#how").hidden = name !== "intake";
  window.scrollTo(0, 0);
  const heading = $(`#view-${name} h1, #view-${name} h2`);
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }
}

function showError(message) {
  const box = $("#form-error");
  box.textContent = message;
  box.hidden = false;
}
function hideError() {
  $("#form-error").hidden = true;
}

// --------------------------------------------------------------- health
async function loadHealth() {
  const pill = $("#model-status");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.llm_configured) {
      pill.dataset.state = "ready";
      pill.textContent = "Agents ready";
      pill.title = `${data.provider}, ${data.model}`;
    } else {
      pill.dataset.state = "missing";
      pill.textContent = "No model key on the server";
      pill.title = "Set GEMINI_API_KEY in the server environment";
    }
  } catch {
    pill.dataset.state = "missing";
    pill.textContent = "Server unreachable";
  }
}

// ----------------------------------------------------------------- file
function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
}

function setFile(file) {
  const sheet = $("#dropzone");
  const title = $("#sheet-title");
  const hint = $("#sheet-hint");
  if (!file) {
    state.file = null;
    $("#resume-input").value = "";
    sheet.classList.remove("has-file");
    title.textContent = "Drop your resume here";
    hint.replaceChildren("PDF, up to 4 MB. Or ", h("u", {}, "choose a file"), ".");
    $("#resume-clear").hidden = true;
    return;
  }
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  if (!isPdf) return showError("That file isn’t a PDF. Export your resume as a PDF and try again.");
  if (file.size > MAX_PDF_BYTES) return showError("That PDF is larger than 4 MB. Export a smaller version and try again.");
  hideError();
  state.file = file;
  sheet.classList.add("has-file");
  title.textContent = file.name;
  hint.textContent = `PDF, ${formatSize(file.size)}. Click to replace.`;
  $("#resume-clear").hidden = false;
}

function wireDropzone() {
  const sheet = $("#dropzone");
  const input = $("#resume-input");
  input.addEventListener("change", () => setFile(input.files[0] || null));
  $("#resume-clear").addEventListener("click", () => setFile(null));
  for (const type of ["dragenter", "dragover"]) {
    sheet.addEventListener(type, (e) => { e.preventDefault(); sheet.classList.add("is-dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    sheet.addEventListener(type, () => sheet.classList.remove("is-dragging"));
  }
  sheet.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });
  // a file dropped outside the zone should not navigate away from the page
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
}

async function useSample(kind) {
  hideError();
  try {
    const [sample, pdf] = await Promise.all([
      fetch("/api/sample").then((r) => r.json()),
      fetch("/api/sample/resume.pdf").then((r) => r.blob()),
    ]);
    setFile(new File([pdf], "sample_resume.pdf", { type: "application/pdf" }));
    $("#job-text").value = kind === "vague" ? sample.vague_job_description : sample.job_description;
    $("#job-url").value = "";
    $("#submit").focus();
  } catch {
    showError("The sample couldn’t be loaded. Check that the server is running.");
  }
}

// --------------------------------------------------------------- submit
async function onSubmit(event) {
  event.preventDefault();
  hideError();
  const jobText = $("#job-text").value.trim();
  const jobUrl = $("#job-url").value.trim();
  if (!state.file) return showError("Add your resume as a PDF first.");
  if (!jobText && !jobUrl) return showError("Paste the job description or add a link to the posting.");
  if (jobUrl && !/^https?:\/\//i.test(jobUrl)) return showError("The job link should start with http:// or https://.");

  const button = $("#submit");
  button.disabled = true;
  button.textContent = "Starting…";
  const body = new FormData();
  body.append("resume", state.file);
  body.append("job_description", jobText);
  body.append("job_url", jobUrl);
  let res;
  try {
    res = await fetch("/api/analyze", { method: "POST", body });
  } catch {
    button.disabled = false;
    button.textContent = "Check my fit";
    return showError("Couldn’t reach HireLens. Check your connection and try again.");
  }
  button.disabled = false;
  button.textContent = "Check my fit";
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    const fallback = res.status === 413
      ? "That PDF is too large to upload. Export a smaller version (under 4 MB) and try again."
      : `HireLens returned an error (${res.status}). Try again.`;
    return showError(data?.error?.message || fallback);
  }
  streamRun(res);
}

// ------------------------------------------------------------ live run
function resetRun() {
  state.events = [];
  state.seen = new Set();
  state.active = null;
  state.errored = new Set();
  $("#log").replaceChildren();
  $("#run-title").textContent = "Reading your resume and the job";
  $("#run-status").textContent = "Starting the agents…";
  for (const node of $$("#agent-map .node")) {
    node.classList.remove("is-active", "is-done", "is-error");
    $(".node-note", node).textContent = "";
  }
  for (const edge of $$("#agent-map .edge")) edge.classList.remove("is-live", "is-used");
}

// The analysis runs inside the POST request, and the server streams Server-Sent
// Events back on the same connection. EventSource only does GET, so the stream is
// read with fetch and parsed here. One request means it also works on serverless hosts.
async function streamRun(res) {
  state.runId = null;
  state.result = null;
  resetRun();
  showView("run");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  const handle = (block) => {
    let event = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return;
    const payload = JSON.parse(data.join("\n"));
    if (event === "started") state.runId = payload.run_id;
    else if (event === "trace") onTrace(payload);
    else if (event === "done") { finished = true; onDone(payload); }
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        handle(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
      }
    }
    if (buffer.trim()) handle(buffer);
  } catch {
    // connection dropped; reported below
  }
  if (!finished) {
    renderFailure("The connection to HireLens closed before the analysis finished. "
      + "The server may have hit its time limit. Try again, ideally with the job description pasted in.");
  }
}

function humanize(text) {
  return text.replace(/\b(orchestrator|resume_parser|job_analyst|matcher|critic|career_coach|finalize)\b/g,
    (id) => AGENT_NAMES[id]);
}

function onTrace(ev) {
  if (state.seen.has(ev.step)) return;
  state.seen.add(ev.step);
  state.events.push(ev);
  appendLog(ev);
  updateMap(ev);
}

function appendLog(ev) {
  if (ev.event === "input") return;
  const text = ev.event === "final" ? "Report ready." : humanize(ev.summary);
  const item = h("li", {
    dataset: { agent: ev.agent },
    class: `${QUIET_EVENTS.has(ev.event) ? "is-quiet " : ""}is-${ev.event}`,
  },
  h("time", {}, `${ev.elapsed_s.toFixed(1)}s`),
  QUIET_EVENTS.has(ev.event) ? null : h("span", { class: "who" }, AGENT_NAMES[ev.agent] || ev.agent),
  h("span", { class: "what" }, text));
  const log = $("#log");
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  log.append(item);
  if (nearBottom) log.scrollTop = log.scrollHeight;
}

function node(agent) {
  return $(`#agent-map .node[data-agent="${agent}"]`);
}
function setNodeState(agent, cls) {
  const n = node(agent);
  if (!n) return;
  n.classList.remove("is-active", "is-done", "is-error");
  if (cls) n.classList.add(cls);
}
function note(agent, text) {
  const n = node(agent);
  if (n) $(".node-note", n).textContent = text;
}

function activate(agent) {
  if (state.active === agent || state.errored.has(agent)) return;
  if (state.active && !state.errored.has(state.active)) setNodeState(state.active, "is-done");
  state.active = agent;
  setNodeState(agent, "is-active");
  if (ACTIVITY[agent]) $("#run-status").textContent = `${ACTIVITY[agent]}…`;
}

function updateMap(ev) {
  const { agent, event, data = {} } = ev;
  if (event === "handoff") {
    const key = `${data.from}>${data.to}`;
    for (const edge of $$("#agent-map .edge.is-live")) {
      edge.classList.remove("is-live");
      edge.classList.add("is-used");
    }
    const edge = $(`#agent-map .edge[data-edge="${key}"]`);
    if (edge) edge.classList.add("is-live");
    if (data.to === "finalize") {
      if (state.active && !state.errored.has(state.active)) setNodeState(state.active, "is-done");
      state.active = null;
      $("#run-status").textContent = "Putting your report together…";
    } else if (data.to) {
      activate(data.to);
    }
    return;
  }
  if (event === "final") return;
  activate(agent);

  if (event === "error") {
    state.errored.add(agent);
    setNodeState(agent, "is-error");
    note(agent, "failed");
  } else if (event === "critique") {
    const round = data.round ?? "";
    const verdict = data.critique?.verdict === "approve" ? "approved" : "sent back";
    note("critic", `round ${round}: ${verdict}`);
    if (verdict === "sent back") $("#run-status").textContent = "The critic found unsupported claims and sent the scores back…";
  } else if (event === "revision") {
    note("matcher", "revising");
    $("#run-status").textContent = "The matcher is revising its scores…";
  } else if (event === "message" && agent === "matcher") {
    note("matcher", `round ${data.round ?? 1}`);
  } else if (event === "tool_call" && agent === "job_analyst") {
    const tool = data.tool;
    note("job_analyst", tool === "web_search" ? "searching the web" : tool === "fetch_url" ? "reading a page" : "");
    $("#run-status").textContent = tool === "web_search"
      ? "The job post is thin, so the analyst is searching for the role’s requirements…"
      : "Reading the job posting…";
  } else if (event === "plan" && agent === "orchestrator" && ev.summary.startsWith("Re-plan")) {
    note("orchestrator", "re-planning");
  } else if (event === "message" && agent === "job_analyst" && data.job_profile?.enrichment_used) {
    note("job_analyst", "used the web");
  }
}

function onDone(payload) {
  state.result = payload;
  if (payload.status === "failed" || !payload.report) {
    renderFailure(payload.error || "The analysis stopped unexpectedly.");
  } else {
    renderReport(payload.report);
  }
}

// --------------------------------------------------------------- report
function bandOf(verdict = "") {
  const v = verdict.toLowerCase();
  if (v.startsWith("strong")) return "strong";
  if (v.startsWith("moderate")) return "moderate";
  if (v.startsWith("weak")) return "weak";
  return "poor";
}

function fileLink(label, filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  return h("a", { class: "button-secondary", href: url, download: filename }, label);
}

// The report and trace arrive with the final "done" event, so downloads are
// built in the browser and never depend on the server remembering the run.
function downloads() {
  const r = state.result;
  if (!r) return [];
  const id = r.run_id || "run";
  return [
    r.report ? fileLink("Download report", `hirelens_report_${id}.json`, JSON.stringify(r.report, null, 2), "application/json") : null,
    r.trace ? fileLink("Download trace (JSON)", `hirelens_trace_${id}.json`, JSON.stringify(r.trace, null, 2), "application/json") : null,
    r.trace_markdown ? fileLink("Download trace (Markdown)", `hirelens_trace_${id}.md`, r.trace_markdown, "text/markdown") : null,
  ].filter(Boolean);
}

function againButton(label) {
  return h("button", {
    type: "button", class: "button-secondary",
    onclick: () => { showView("intake"); $("#job-text").focus(); },
  }, label);
}

function renderFailure(message) {
  const view = $("#view-report");
  fill(view, h("div", { class: "aborted" },
    h("h2", {}, "The analysis didn’t finish"),
    h("p", {}, message),
    h("p", {}, "Nothing was saved. You can try again; if the model is busy, wait a minute first."),
    h("div", { class: "report-foot" }, againButton("Try again"), downloads())));
  showView("report");
}

function renderReport(r) {
  const view = $("#view-report");
  if (r.status === "aborted") {
    fill(view, h("div", { class: "aborted" },
      h("h2", { id: "verdict" }, "We stopped before scoring"),
      h("p", {}, r.message),
      r.warnings?.length ? h("div", { class: "warnings" }, r.warnings.map((w) => h("p", {}, w))) : null,
      h("div", { class: "report-foot" }, againButton("Change the inputs"), downloads())));
    showView("report");
    return;
  }

  const role = [r.job_title || "this role", r.company ? `at ${r.company}` : ""].join(" ").trim();
  const rounds = r.critic_rounds || 0;
  const verification = r.critic_approved
    ? h("p", { class: "verification", dataset: { state: "approved" } },
        h("span", { class: "mark" }, "Verified."),
        h("span", {}, rounds > 1
          ? `The critic checked every claim against your resume and sent the scores back ${rounds - 1} time${rounds > 2 ? "s" : ""} before approving them.`
          : "The critic checked every claim against your resume and approved the scores."))
    : h("p", { class: "verification", dataset: { state: "arbitrated" } },
        h("span", { class: "mark" }, rounds ? "Settled." : "Unverified."),
        h("span", {}, r.message));

  fill(view,
    h("div", { class: "verdict-block" },
      h("div", {},
        h("h2", { class: "verdict-title", id: "verdict", dataset: { band: bandOf(r.verdict) } }, r.verdict),
        h("p", { class: "verdict-role" }, `for ${role}`)),
      h("p", { class: "verdict-score" }, h("strong", {}, r.overall_score), h("span", {}, "out of 100")),
      verification),
    r.warnings?.length ? h("div", { class: "warnings", role: "note" }, r.warnings.map((w) => h("p", {}, w))) : null,
    contribution(r),
    sectionRows(r),
    fitLists(r),
    coachTabs(r),
    h("div", { class: "report-foot" }, againButton("Check another job"), h("span", { class: "spacer" }), downloads()),
  );
  showView("report");
}

function contribution(r) {
  const order = Object.keys(r.weights || {});
  const rawTotal = order.reduce((sum, k) => sum + (r.weights[k] * (r.section_scores[k]?.score ?? 0)), 0);
  const capped = Math.round(rawTotal) > r.overall_score;
  const bar = h("div", { class: "contrib-bar", role: "img",
    "aria-label": order.map((k) => `${SECTION_NAMES[k]} ${r.section_scores[k]?.score} of 100, weight ${Math.round(r.weights[k] * 100)}%`).join("; ") });
  const labels = h("div", { class: "contrib-labels", "aria-hidden": "true" });
  for (const k of order) {
    const grow = r.weights[k] * 100;
    const score = r.section_scores[k]?.score ?? 0;
    bar.append(h("span", { class: "contrib-seg", style: `flex: ${grow} 1 0` }, h("i", { style: `width: ${score}%` })));
    labels.append(h("span", { style: `flex: ${grow} 1 0` }, h("b", {}, score), SECTION_NAMES[k]));
  }
  return h("div", { class: "contrib" }, bar, labels,
    h("p", { class: "contrib-note" },
      "Each block is sized by how much that section counts towards the overall score; the filled part is your score in it.",
      capped ? " The overall score is capped at 55 because more than half of the required skills are missing." : ""));
}

function annotationsFor(r, section) {
  const notes = [];
  for (const round of r.critic_history || []) {
    if (round.verdict !== "revise") continue;
    for (const issue of round.issues || []) {
      if (issue.section === section && issue.problem && !notes.includes(issue.problem)) notes.push(issue.problem);
    }
  }
  return notes.slice(0, 2);
}

function sectionRows(r) {
  const list = h("div", { class: "sections" });
  for (const [key, sec] of Object.entries(r.section_scores)) {
    const history = r.score_history?.[key] || [sec.score];
    const changed = history.length > 1 && history[0] !== sec.score;
    const weight = Math.round((r.weights?.[key] || 0) * 100);
    list.append(h("div", { class: "section-row" },
      h("div", { class: "section-name" }, SECTION_NAMES[key] || key, h("small", {}, `${weight}% of the score`)),
      h("div", { class: "section-score", "aria-label": changed ? `${sec.score}, revised from ${history[0]}` : `${sec.score}` },
        changed ? h("s", { "aria-hidden": "true" }, history[0]) : null, sec.score),
      h("div", { class: "section-detail" },
        h("p", {}, sec.reasoning),
        changed ? annotationsFor(r, key).map((t) => h("p", { class: "annotation" }, t)) : null,
        sec.evidence?.length
          ? h("ul", { class: "evidence" }, sec.evidence.map((ev) => h("li", {},
              r.critic_approved ? h("span", { class: "checked" }, "Found in your resume:") : null,
              h("q", {}, ev.resume_quote), ` ${ev.claim ? `(${ev.claim})` : ""}`)))
          : null)));
  }
  return list;
}

function fitLists(r) {
  const have = [...new Set([...(r.matched_must_have || []), ...(r.strengths || [])])];
  const missing = [
    ...(r.skill_gaps || []).map((g) => h("li", {}, g, h("span", { class: "tag" }, "required"))),
    ...(r.nice_to_have_gaps || []).map((g) => h("li", { class: "nice" }, g, h("span", { class: "tag" }, "nice to have"))),
  ];
  return h("div", { class: "fit-lists" },
    h("div", { class: "have" }, h("h3", {}, "What you already bring"),
      have.length ? h("ul", {}, have.map((x) => h("li", {}, x))) : h("p", { class: "empty" }, "No clear matches found.")),
    h("div", { class: "missing" }, h("h3", {}, "What’s missing"),
      missing.length ? h("ul", {}, missing) : h("p", { class: "empty" }, "No gaps found for this role.")));
}

function coachTabs(r) {
  const c = r.coaching;
  const tabs = [
    ["bullets", "Resume bullets", () => bulletsPanel(c)],
    ["questions", "Interview questions", () => questionsPanel(c)],
    ["gaps", "Close the gaps", () => gapsPanel(c)],
    ["agents", "How the agents got here", () => agentsPanel(r)],
  ];
  const tablist = h("div", { class: "tablist", role: "tablist", "aria-label": "Suggestions and details" });
  const panels = [];
  tabs.forEach(([id, label, build], i) => {
    const tab = h("button", {
      type: "button", class: "tab", role: "tab", id: `tab-${id}`, "aria-controls": `panel-${id}`,
      "aria-selected": i === 0 ? "true" : "false", tabindex: i === 0 ? "0" : "-1",
    }, label);
    const panel = h("div", { class: "panel", role: "tabpanel", id: `panel-${id}`, "aria-labelledby": `tab-${id}`, tabindex: "0" }, build());
    panel.hidden = i !== 0;
    tablist.append(tab);
    panels.push(panel);
  });
  const select = (index) => {
    $$(".tab", tablist).forEach((t, i) => {
      t.setAttribute("aria-selected", String(i === index));
      t.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
    });
    $$(".tab", tablist)[index].focus();
  };
  tablist.addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab) select($$(".tab", tablist).indexOf(tab));
  });
  tablist.addEventListener("keydown", (e) => {
    const tabsEls = $$(".tab", tablist);
    const current = tabsEls.indexOf(document.activeElement);
    if (current < 0) return;
    const next = { ArrowRight: current + 1, ArrowLeft: current - 1, Home: 0, End: tabsEls.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    select((next + tabsEls.length) % tabsEls.length);
  });
  return h("div", { class: "coach" }, tablist, panels);
}

const noCoaching = () => h("p", { class: "empty" }, "Suggestions couldn’t be generated for this run. The scores above are complete.");

function bulletsPanel(c) {
  if (!c?.bullet_suggestions?.length) return noCoaching();
  return h("div", {},
    h("p", { class: "empty", style: "margin-bottom:1.25rem" },
      "Numbers in [brackets] are placeholders. Fill them in only with figures that are true."),
    h("div", { class: "rewrite-head", "aria-hidden": "true" }, h("span", {}, "Your bullet"), h("span", {}, "Rewritten for this role")),
    h("div", { class: "rewrites" }, c.bullet_suggestions.map((b) => h("div", { class: "rewrite" },
      h("p", { class: `before${b.original ? "" : " is-new"}` }, b.original || "New bullet"),
      h("p", { class: "after" }, b.improved),
      b.targets ? h("p", { class: "targets" }, `Helps with: ${b.targets}`) : null))));
}

function questionsPanel(c) {
  if (!c?.interview_questions?.length) return noCoaching();
  const label = (cat = "") => cat.charAt(0).toUpperCase() + cat.slice(1);
  return h("ol", { class: "questions" }, c.interview_questions.map((q) => h("li", {},
    h("p", { class: "q" }, q.question),
    h("p", { class: "why" }, h("span", { class: "cat" }, label(q.category)), q.why_asked))));
}

function gapsPanel(c) {
  if (!c?.gap_actions?.length) return noCoaching();
  return h("dl", { class: "actions-list" }, c.gap_actions.map((a) => h("div", {}, h("dt", {}, a.gap), h("dd", {}, a.action))));
}

function agentsPanel(r) {
  const rounds = (r.critic_history || []).map((round) => h("li", { dataset: { verdict: round.verdict } },
    h("p", { class: "round-head" }, `Critic, round ${round.round}: ${round.verdict === "approve" ? "approved" : "sent back to the matcher"}`),
    round.summary ? h("p", { class: "round-summary" }, round.summary) : null,
    round.issues?.length ? h("ul", {}, round.issues.map((i) => h("li", {}, `${SECTION_NAMES[i.section] || i.section}: ${i.problem}`))) : null));

  const filter = h("select", { id: "trace-filter" },
    h("option", { value: "" }, "All agents"),
    Object.entries(AGENT_NAMES).filter(([k]) => k !== "finalize").map(([k, v]) => h("option", { value: k }, v)));
  const count = h("span", { class: "empty" });
  const tbody = h("tbody");
  const fill = () => {
    const rows = state.events.filter((e) => !filter.value || e.agent === filter.value);
    count.textContent = `${rows.length} of ${state.events.length} events`;
    tbody.replaceChildren(...rows.map((e) => h("tr", { dataset: { agent: e.agent } },
      h("td", {}, e.step), h("td", {}, `${e.elapsed_s.toFixed(1)}s`),
      h("td", { class: "agent-cell" }, AGENT_NAMES[e.agent] || e.agent), h("td", {}, e.event.replace("_", " ")),
      h("td", {}, humanize(e.summary)))));
  };
  filter.addEventListener("change", fill);
  fill();

  return h("div", {},
    rounds.length ? h("ol", { class: "rounds" }, rounds) : null,
    r.job_sources?.length ? h("p", { class: "empty", style: "margin-bottom:1.5rem" },
      `Job requirements came from: ${r.job_sources.join("; ")}.`) : null,
    h("div", { class: "trace-tools" }, h("label", { for: "trace-filter" }, "Show"), filter, count),
    h("div", { class: "trace-table-wrap" }, h("table", { class: "trace-table" },
      h("thead", {}, h("tr", {}, ["#", "Time", "Agent", "Event", "What happened"].map((t) => h("th", { scope: "col" }, t)))),
      tbody)));
}

// ----------------------------------------------------------------- boot
wireDropzone();
$("#intake-form").addEventListener("submit", onSubmit);
for (const btn of $$("[data-sample]")) btn.addEventListener("click", () => useSample(btn.dataset.sample));
loadHealth();
