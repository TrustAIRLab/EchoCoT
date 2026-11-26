"use strict";

const $ = id => document.getElementById(id);
const node = (tag, className = "", text = "") => {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
};
const fmt = value => Number.isFinite(value) ? value.toFixed(3) : "N/A";
const integer = value => Number.isFinite(value) ? value.toLocaleString() : "—";
let catalog, state, drafts, pending = false, connected = false;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request failed.");
  return result;
}

function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}

function savedDrafts() {
  return { injections: state.config.injections.map(i => i.injection), final: state.config.final_instruction };
}

function rememberDrafts() {
  sessionStorage.setItem(`echocot-demo:${state.config.key}`, JSON.stringify(drafts));
}

function applyState(next) {
  if (state && next.revision <= state.revision) return;
  const newConversation = state?.config?.id !== next.config?.id;
  state = next;
  if (!state.config) return;
  if (newConversation) {
    drafts = savedDrafts();
    const saved = sessionStorage.getItem(`echocot-demo:${state.config.key}`);
    if (saved) {
      const previous = JSON.parse(saved);
      drafts.injections = drafts.injections.map((text, i) => previous.injections[i] ?? text);
      drafts.final = previous.final ?? drafts.final;
    }
  }
  $("model").value = state.config.key;
  $("dataset").value = state.config.dataset;
  const count = state.config.sample_count;
  $("model-note").textContent = state.config.notes;
  $("sample-note").textContent = state.config.sample_idx + 1 === count ? "Last sample in this dataset." : "Samples are shown in dataset order.";
  $("question").textContent = state.question;
  $("question-meta").textContent = `${state.config.dataset} · Sample ${state.config.sample_idx + 1} of ${count}`;
  notice(state.error || "");
  renderReference();
  renderProcess();
  renderControls();
}

function renderReference() {
  const response = state.response;
  $("reference-tokens").textContent = `${integer(response?.ground_truth_tokens)} tokens`;
  $("summary-label").hidden = true;
  $("reference-content").textContent = "";
  if (!response) {
    $("reference-note").textContent = state.busy ? "Waiting for the model's initial reasoning…" : "The model's initial reasoning will appear here after you send the question.";
  } else if (response.ground_truth_cot) {
    $("reference-note").textContent = "Reference from the model's initial response.";
    $("reference-content").textContent = response.ground_truth_cot;
  } else {
    $("reference-note").textContent = "This API response does not expose the full ground-truth CoT.";
    $("summary-label").hidden = !response.ground_truth_cot_summary;
    $("reference-content").textContent = response.ground_truth_cot_summary || "No reasoning text was returned.";
  }
}

function metric(label, value, passed) {
  const element = node("span", "", label);
  element.append(node("strong", passed === undefined ? "" : passed ? "passed" : "failed", value));
  return element;
}

function extractionResult(turn) {
  const success = Boolean(turn.scratchpad_content) && Number.isFinite(turn.length_error)
    && turn.length_error <= state.config.length_error_threshold
    && (state.config.length_only || (Number.isFinite(turn.token_em) && turn.token_em >= state.config.token_em_threshold));
  const result = node("div", `extraction-result ${success ? "success" : "failure"}`);
  const icon = node("span", "result-icon", success ? "✓" : "×");
  icon.setAttribute("aria-hidden", "true");
  const successLabel = state.config.length_only ? "(Potential) successful extraction" : "Successful extraction";
  result.append(icon, node("strong", "", success ? successLabel : "Failed extraction"));
  return result;
}

function contentBlock(text, className, heading) {
  const element = node("pre", className, text);
  const expand = node("button", "text-button", "Expand");
  expand.onclick = () => {
    element.classList.toggle("expanded");
    expand.textContent = element.classList.contains("expanded") ? "Collapse" : "Expand";
  };
  heading.append(expand);
  return element;
}

function responseCards(event) {
  const cards = [];
  if (event.status === "running") {
    const waiting = node("div", "response-card waiting"), timer = node("span");
    timer.dataset.timer = event.started_at;
    waiting.append(node("span", "spinner"), node("span", "", "Waiting for the model"), timer);
    return [waiting];
  }
  for (const index of event.turns) {
    const turn = state.response.scratchpads.find(t => t.turn_idx === index);
    const card = node("article", "response-card"), heading = node("div", "card-heading");
    heading.append(node("h3", "", `Scratchpad tool · ${index === 0 ? "initial response" : `Turn ${index}`}`));
    const content = contentBlock(turn.scratchpad_content || "(Empty scratchpad)", "scratchpad", heading);
    const scores = node("div", "metrics");
    scores.append(metric("Tokens / GT", `${integer(turn.scratchpad_tokens)} / ${integer(turn.ground_truth_cot_tokens)}`),
      metric("Length error", fmt(turn.length_error), Number.isFinite(turn.length_error) ? turn.length_error <= state.config.length_error_threshold : undefined),
      metric("Token EM", fmt(turn.token_em), Number.isFinite(turn.token_em) ? turn.token_em >= state.config.token_em_threshold : undefined));
    card.append(heading, content, scores, extractionResult(turn));
    cards.push(card);
  }
  if (event.answer) {
    const card = node("article", "response-card"), heading = node("div", "card-heading");
    heading.append(node("h3", "", "Final answer"));
    card.append(heading, node("pre", "answer", event.answer));
    cards.push(card);
  } else if (event.status === "complete" && !event.turns.length) {
    cards.push(node("p", "empty-note", "The session ended without further text from the extractor."));
  }
  if (event.status === "error") cards.push(node("p", "notice", state.error || "Model request failed."));
  return cards;
}

function renderProcess() {
  const initial = state.events.find(e => e.action === "start");
  if (initial) $("initial-response").replaceChildren(...responseCards(initial));
  else {
    const placeholder = node("div", "initial-placeholder");
    placeholder.append(node("strong", "", "Initial scratchpad"), node("span", "", "Send the question to begin. Then edit and send each injection below."));
    $("initial-response").replaceChildren(placeholder);
  }
  const injectionEvents = state.events.filter(e => e.action !== "start");
  const stages = [...state.config.injections.map(i => ({ label: `Turn ${i.turn_idx} injection`, number: i.turn_idx, action: "inject" })),
    { label: `Turn ${state.config.injections.length + 1} · Final instruction`, number: state.config.injections.length + 1, action: "final" }];
  $("turns").replaceChildren(...stages.map((stage, index) => {
    const event = injectionEvents[index];
    const block = node("div", "turn-block"), card = node("article", "injection-card");
    block.id = `turn-${index}`;
    const heading = node("div", "card-heading"), label = node("h3", "turn-label");
    label.append(node("span", "turn-number", String(stage.number)), document.createTextNode(stage.label));
    heading.append(label);
    card.append(heading);
    if (event) {
      card.append(contentBlock(event.input, "sent-injection", heading));
      block.append(card, ...responseCards(event));
    } else {
      heading.append(node("span", "edit-label", "Feel Free to Edit"));
      const editor = node("textarea", "injection-editor");
      editor.dataset.index = index;
      editor.setAttribute("aria-label", stage.label);
      editor.spellcheck = false;
      editor.value = stage.action === "final" ? drafts.final : drafts.injections[index];
      editor.rows = stage.action === "final" ? 3 : 5;
      editor.placeholder = "Enter your injection for this turn.";
      const actions = node("div", "injection-actions"), send = node("button", "primary send-injection", `Send Turn ${stage.number}`);
      send.dataset.index = index;
      send.dataset.action = stage.action;
      send.onclick = () => advance(stage.action, editor.value);
      editor.oninput = () => {
        if (stage.action === "final") drafts.final = editor.value;
        else drafts.injections[index] = editor.value;
        rememberDrafts(); renderControls();
      };
      actions.append(node("span", "", stage.action === "final" ? "Ends the current conversation." : "Sent as feedback to the scratchpad tool."), send);
      card.append(editor, actions);
      block.append(card);
    }
    return block;
  }));
  $("conversation-end").hidden = state.phase !== "complete" && state.phase !== "error";
  $("conversation-end").textContent = state.phase === "error" ? "The request failed. Restart this conversation or click Next sample." : "Conversation complete. Click Next sample, or restart to try different injections.";
}

function renderControls() {
  const working = pending || state?.busy;
  for (const id of ["model", "dataset", "restart", "restore"]) $(id).disabled = working || !state?.config;
  const sampleCount = state?.config?.sample_count || 0;
  $("next-sample").disabled = working || !state?.config || state.config.sample_idx + 1 >= sampleCount;
  $("start").disabled = working || !connected || state?.next_action !== "start";
  $("start").textContent = state?.events.length ? "Question sent" : "Send question";
  if (state?.config) {
    const turn = state.config.injections[state.sent_injections]?.turn_idx;
    $("phase").textContent = working ? "Model request in progress…" : ({ ready: "Ready to send the selected question", waiting: state.next_action === "inject" ? `Ready for Turn ${turn} injection` : "Ready for the final instruction", complete: "Conversation complete", error: "Request failed" })[state.phase];
    for (const editor of document.querySelectorAll(".injection-editor")) editor.disabled = working;
    for (const button of document.querySelectorAll(".send-injection")) {
      const index = Number(button.dataset.index), final = button.dataset.action === "final";
      const text = final ? drafts.final : drafts.injections[index];
      const available = state.next_action === button.dataset.action && (final || index === state.sent_injections);
      button.disabled = working || !connected || !available || !text.trim();
      button.closest(".injection-card").classList.toggle("active", available && !working);
    }
  }
}

async function configure(event, sampleIdx) {
  event?.preventDefault();
  if (pending || state?.busy || !$("selection").reportValidity()) return;
  const model = $("model").value;
  const dataset = $("dataset").value;
  const index = sampleIdx ?? (state?.config?.key === model && state.config.dataset === dataset ? state.config.sample_idx : 0);
  const data = { model, dataset, sample_idx: index };
  pending = true; renderControls(); notice();
  try { applyState(await api("/api/configure", data)); }
  catch (error) { notice(error.message); }
  finally { pending = false; renderControls(); }
}

async function nextSample() {
  if (!state?.config) return;
  const index = state.config.sample_idx + 1;
  const count = state.config.sample_count;
  if (index < count) await configure(undefined, index);
}

async function advance(action, injection) {
  if (pending || state.busy || state.next_action !== action) return;
  pending = true; renderControls(); notice();
  try { applyState(await api("/api/action", { action, revision: state.revision, injection })); }
  catch (error) { notice(error.message); }
  finally { pending = false; renderControls(); }
}

async function init() {
  catalog = await api("/api/catalog");
  for (const model of catalog.models) $("model").add(new Option(model.key, model.key));
  for (const dataset of catalog.datasets) $("dataset").add(new Option(dataset.name, dataset.name));
  $("model").value = catalog.default_model;
  if (catalog.datasets.some(d => d.name === "OpenThoughts")) $("dataset").value = "OpenThoughts";
  $("selection").onsubmit = configure;
  $("model").onchange = configure;
  $("dataset").onchange = configure;
  $("next-sample").onclick = nextSample;
  $("start").onclick = () => advance("start");
  $("restore").onclick = () => { drafts = savedDrafts(); rememberDrafts(); renderProcess(); renderControls(); };
  const initial = await api("/api/state");
  if (initial.config) applyState(initial);
  else await configure();
  const events = new EventSource("/api/events");
  events.addEventListener("state", event => {
    connected = true;
    $("connection").textContent = "Connected";
    $("connection").className = "connection connected";
    applyState(JSON.parse(event.data)); renderControls();
  });
  events.onerror = () => { connected = false; $("connection").textContent = "Reconnecting"; $("connection").className = "connection"; renderControls(); };
  setInterval(() => {
    for (const element of document.querySelectorAll("[data-timer]")) element.textContent = `· ${Math.floor(Date.now() / 1000 - Number(element.dataset.timer))}s`;
  }, 500);
}

init().catch(error => notice(error.message));
