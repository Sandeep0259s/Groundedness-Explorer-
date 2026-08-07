const el = (id) => document.getElementById(id);

const chat = el("chat");
const emptyState = el("empty-state");
const composer = el("composer");
const questionInput = el("question-input");
const sendBtn = el("send-btn");
const labelGroups = el("label-groups");
const labelCreateForm = el("label-create-form");
const newLabelInput = el("new-label-input");
const uploadLabelSelect = el("upload-label-select");
const askLabelSelect = el("ask-label-select");
const fileInput = el("file-input");
const dropzone = el("dropzone");
const reingestBtn = el("reingest-btn");
const topkInput = el("topk");
const topkValue = el("topk-value");
const statusDot = el("status-dot");
const statusText = el("status-text");
const themeToggle = el("theme-toggle");
const themeIcon = el("theme-icon");
const sidebarToggle = el("sidebar-toggle");
const appRoot = document.querySelector(".app");
const toastStack = el("toast-stack");
const messageTemplate = el("message-template");
const answerTemplate = el("answer-template");
const ingestBadge = el("ingest-badge");
const newChatBtn = el("new-chat-btn");
const modelSelect = el("model-select");
const modelRefreshBtn = el("model-refresh-btn");
const visionModelSelect = el("vision-model-select");
const visionModelHint = el("vision-model-hint");
const deviceSelect = el("device-select");
const performanceSelect = el("performance-select");
const statCpuValue = el("stat-cpu-value");
const statCpuBar = el("stat-cpu-bar");
const statRamValue = el("stat-ram-value");
const statRamBar = el("stat-ram-bar");
const statGpuRow = el("stat-gpu-row");
const statGpuLabel = el("stat-gpu-label");
const statGpuValue = el("stat-gpu-value");
const statGpuBar = el("stat-gpu-bar");

let hasDocuments = false;
let conversationId = null;

/* ---------- theme ---------- */
function applyTheme(theme) {
  if (theme) {
    document.documentElement.setAttribute("data-theme", theme);
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  const isDark =
    theme === "dark" ||
    (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  themeIcon.textContent = isDark ? "☀" : "☾";
}

(function initTheme() {
  const saved = localStorage.getItem("theme");
  applyTheme(saved);
})();

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const isDark =
    current === "dark" ||
    (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const next = isDark ? "light" : "dark";
  localStorage.setItem("theme", next);
  applyTheme(next);
});

/* ---------- sidebar (mobile) ---------- */
sidebarToggle.addEventListener("click", () => {
  appRoot.classList.toggle("sidebar-open");
});

/* ---------- toasts ---------- */
function showToast(message, kind = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  toastStack.appendChild(toast);
  setTimeout(() => toast.remove(), 4500);
}

/* ---------- backend status ---------- */
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    statusDot.className = "status-dot online";
    statusText.textContent = "Backend online";
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Backend unreachable";
  }
}

/* ---------- labels + documents ---------- */
function basename(path) {
  return path.split(/[\\/]/).pop();
}

// Labels are the organizing unit; each carries its own document list. A
// single /api/labels call gives us both — label metadata and, since sources
// already know their label, everything needed to group them client-side.
async function refreshLabels() {
  try {
    const [labelsRes, docsRes] = await Promise.all([
      fetch("/api/labels"),
      fetch("/api/documents"),
    ]);
    const { labels } = await labelsRes.json();
    const { sources } = await docsRes.json();
    renderLabelGroups(labels, sources);
    populateLabelSelects(labels);
    hasDocuments = sources.length > 0;
    if (hasDocuments) {
      emptyState.querySelector("h2").textContent = "Ready when you are";
      emptyState.querySelector("p").textContent = "Ask a question about your documents below.";
    }
  } catch (err) {
    console.error(err);
  }
}

function populateLabelSelects(labels) {
  const previousUpload = uploadLabelSelect.value;
  const previousAsk = askLabelSelect.value;

  uploadLabelSelect.innerHTML = labels
    .map((l) => `<option value="${l.name}">${l.name}</option>`)
    .join("");
  if (labels.some((l) => l.name === previousUpload)) uploadLabelSelect.value = previousUpload;

  askLabelSelect.innerHTML =
    `<option value="">All labels</option>` +
    labels.map((l) => `<option value="${l.name}">${l.name}</option>`).join("");
  if (labels.some((l) => l.name === previousAsk)) askLabelSelect.value = previousAsk;
}

function renderLabelGroups(labels, sources) {
  const byLabel = {};
  for (const s of sources) {
    (byLabel[s.label] ||= []).push(s);
  }

  labelGroups.innerHTML = labels
    .map((label) => {
      const docs = byLabel[label.name] || [];
      const isProtected = label.name === "general";
      const docsHtml = docs.length
        ? docs
            .map(
              (d) => `
              <li class="doc-item">
                <span class="doc-item-name" title="${d.source}">${basename(d.source)}</span>
                <div class="doc-item-actions">
                  <span class="doc-item-chunks">${d.chunks} chunks</span>
                  <button class="doc-delete-btn" data-action="delete-doc" data-source="${encodeURIComponent(d.source)}" title="Remove file">×</button>
                </div>
              </li>`
            )
            .join("")
        : `<p class="label-group-empty">No files yet.</p>`;

      return `
        <div class="label-group" data-label="${label.name}">
          <div class="label-group-header">
            <span class="label-name">
              <span class="label-name-text">${label.name}</span>
              ${label.ephemeral ? '<span class="label-ephemeral-tag">session</span>' : ""}
            </span>
            <div class="label-actions">
              <span class="label-count">${label.document_count} · ${label.chunk_count}c</span>
              <button class="icon-btn" data-action="clear-label" data-label="${label.name}" title="Clear all files in this label">⟲</button>
              ${
                isProtected
                  ? ""
                  : `<button class="icon-btn" data-action="delete-label" data-label="${label.name}" title="Delete label">🗑</button>`
              }
            </div>
          </div>
          <ul class="doc-list">${docsHtml}</ul>
        </div>`;
    })
    .join("");
}

labelGroups.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-action]");
  if (!btn) return;
  const { action } = btn.dataset;

  if (action === "delete-doc") {
    const source = decodeURIComponent(btn.dataset.source);
    if (!confirm(`Remove "${basename(source)}" from the index?`)) return;
    try {
      await fetch(`/api/documents?source=${encodeURIComponent(source)}`, { method: "DELETE" });
      showToast(`Removed ${basename(source)}`, "success");
      refreshLabels();
    } catch (err) {
      showToast("Failed to remove file", "error");
      console.error(err);
    }
  } else if (action === "delete-label") {
    const label = btn.dataset.label;
    if (!confirm(`Delete label "${label}" and everything in it? This can't be undone.`)) return;
    try {
      const res = await fetch(`/api/labels/${encodeURIComponent(label)}`, { method: "DELETE" });
      if (!res.ok) throw new Error((await res.json()).detail || "delete failed");
      showToast(`Deleted label "${label}"`, "success");
      refreshLabels();
    } catch (err) {
      showToast(`Could not delete label: ${err.message}`, "error");
    }
  } else if (action === "clear-label") {
    const label = btn.dataset.label;
    if (!confirm(`Clear all files in "${label}"? This can't be undone.`)) return;
    try {
      await fetch(`/api/labels/${encodeURIComponent(label)}/clear`, { method: "POST" });
      showToast(`Cleared "${label}"`, "success");
      refreshLabels();
    } catch (err) {
      showToast("Failed to clear label", "error");
      console.error(err);
    }
  }
});

labelCreateForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = newLabelInput.value.trim();
  if (!name) return;
  try {
    const res = await fetch("/api/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "could not create label");
    newLabelInput.value = "";
    showToast(`Created label "${name}"`, "success");
    refreshLabels();
  } catch (err) {
    showToast(err.message, "error");
  }
});

let activeIngestJobs = 0;

function setIngesting(isIngesting) {
  activeIngestJobs = Math.max(0, activeIngestJobs + (isIngesting ? 1 : -1));
  ingestBadge.hidden = activeIngestJobs === 0;
}

// Ingestion runs as a background job on the server so large files don't block
// the app — existing documents stay fully queryable while it runs. This polls
// until the job finishes, then refreshes the document list automatically.
function pollJob(jobId, { onDone, onError }) {
  const tick = async () => {
    let job;
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error(await res.text());
      job = await res.json();
    } catch (err) {
      onError(err);
      return;
    }

    if (job.status === "done") {
      onDone(job);
    } else if (job.status === "error") {
      onError(new Error(job.error || "ingestion failed"));
    } else {
      setTimeout(tick, 1200);
    }
  };
  tick();
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList).filter((f) =>
    /\.(pdf|txt|md|docx|csv|xlsx|html?|png|jpe?g|bmp|tiff?|webp|mp4|mov|avi|mkv|webm|mp3|wav|m4a|flac|ogg)$/i.test(f.name)
  );
  if (files.length === 0) {
    showToast("That file type isn't supported", "error");
    return;
  }

  const label = uploadLabelSelect.value || "general";
  const formData = new FormData();
  formData.append("label", label);
  files.forEach((f) => formData.append("files", f));

  setIngesting(true);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const { job_id, skipped } = await res.json();

    if (skipped && skipped.length) {
      showToast(`Skipped ${skipped.length} unsupported/oversized file(s)`, "error");
    }
    showToast(`Ingesting ${files.length} file(s) into "${label}" in the background — you can keep asking questions`, "info");

    pollJob(job_id, {
      onDone: (job) => {
        const totalChunks = (job.results || []).reduce((sum, r) => sum + (r.chunks || 0), 0);
        showToast(`Ingested ${totalChunks} chunks from ${files.length} file(s)`, "success");
        refreshLabels();
        setIngesting(false);
      },
      onError: (err) => {
        showToast("Ingestion failed — see console for details", "error");
        console.error(err);
        setIngesting(false);
      },
    });
  } catch (err) {
    showToast("Upload failed — see console for details", "error");
    console.error(err);
    setIngesting(false);
  }
}

fileInput.addEventListener("change", (e) => uploadFiles(e.target.files));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

reingestBtn.addEventListener("click", async () => {
  reingestBtn.disabled = true;
  setIngesting(true);
  try {
    const res = await fetch("/api/ingest", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    showToast("Ingesting data/raw in the background — you can keep asking questions", "info");

    pollJob(job_id, {
      onDone: (job) => {
        showToast(`Ingested ${job.chunks_ingested} chunks from data/raw`, "success");
        refreshLabels();
        setIngesting(false);
        reingestBtn.disabled = false;
      },
      onError: (err) => {
        showToast("Ingest failed — see console for details", "error");
        console.error(err);
        setIngesting(false);
        reingestBtn.disabled = false;
      },
    });
  } catch (err) {
    showToast("Ingest failed — see console for details", "error");
    console.error(err);
    setIngesting(false);
    reingestBtn.disabled = false;
  }
});

/* ---------- settings ---------- */
topkInput.addEventListener("input", () => {
  topkValue.textContent = topkInput.value;
});

/* ---------- performance: device, CPU/GPU usage, live resource panel ---------- */
async function refreshDevices() {
  try {
    const res = await fetch("/api/system/devices");
    const { devices, active } = await res.json();
    deviceSelect.innerHTML = devices.map((d) => `<option value="${d.id}">${d.name}</option>`).join("");
    deviceSelect.value = active;
  } catch (err) {
    console.error(err);
  }
}

deviceSelect.addEventListener("change", async () => {
  const device = deviceSelect.value;
  deviceSelect.disabled = true;
  try {
    const res = await fetch("/api/system/device", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "failed to switch device");
    showToast(`Switched to ${device.toUpperCase()} — local models reloaded`, "success");
  } catch (err) {
    showToast(`Could not switch device: ${err.message}`, "error");
    refreshDevices();
  } finally {
    deviceSelect.disabled = false;
  }
});

async function refreshPerformanceMode() {
  try {
    const res = await fetch("/api/system/performance");
    const { mode } = await res.json();
    performanceSelect.value = mode;
  } catch (err) {
    console.error(err);
  }
}

performanceSelect.addEventListener("change", async () => {
  const mode = performanceSelect.value;
  try {
    const res = await fetch("/api/system/performance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "failed to change performance mode");
    showToast(`CPU usage set to "${mode}"`, "success");
  } catch (err) {
    showToast(`Could not change performance mode: ${err.message}`, "error");
  }
});

// Every model comes straight from Ollama's own metadata (name + capability
// list) — nothing here is a hardcoded model name, so a newly pulled or
// `ollama create`-d model just shows up after a refresh.
async function refreshModels() {
  modelRefreshBtn.disabled = true;
  try {
    const res = await fetch("/api/system/models");
    if (!res.ok) throw new Error((await res.json()).detail || "could not reach Ollama");
    const { models, active } = await res.json();

    modelSelect.innerHTML = models.map((m) => `<option value="${m.name}">${m.name}</option>`).join("");
    if (active.chat && !models.some((m) => m.name === active.chat)) {
      modelSelect.insertAdjacentHTML("afterbegin", `<option value="${active.chat}">${active.chat} (not pulled)</option>`);
    }
    if (active.chat) modelSelect.value = active.chat;

    const visionModels = models.filter((m) => m.capabilities.includes("vision"));
    visionModelHint.hidden = visionModels.length > 0;
    if (visionModels.length === 0) {
      visionModelSelect.innerHTML = `<option value="">None available</option>`;
      visionModelSelect.disabled = true;
    } else {
      visionModelSelect.disabled = false;
      visionModelSelect.innerHTML = visionModels.map((m) => `<option value="${m.name}">${m.name}</option>`).join("");
      if (active.vision) visionModelSelect.value = active.vision;
    }
  } catch (err) {
    showToast(`Could not list models: ${err.message}`, "error");
  } finally {
    modelRefreshBtn.disabled = false;
  }
}

async function setActiveModel(role, model, selectEl) {
  try {
    const res = await fetch("/api/system/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, role }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "failed to switch model");
    showToast(`${role === "vision" ? "Vision" : "Answering"} model set to "${model}"`, "success");
  } catch (err) {
    showToast(`Could not switch model: ${err.message}`, "error");
    refreshModels();
  }
}

modelSelect.addEventListener("change", () => setActiveModel("chat", modelSelect.value));
visionModelSelect.addEventListener("change", () => setActiveModel("vision", visionModelSelect.value));
modelRefreshBtn.addEventListener("click", refreshModels);

// Polls actual CPU/RAM/GPU load so the user can tell when it's worth raising
// or lowering the performance mode, instead of guessing.
async function refreshStats() {
  try {
    const res = await fetch("/api/system/stats");
    if (!res.ok) return;
    const stats = await res.json();

    statCpuValue.textContent = `${Math.round(stats.cpu.percent)}%`;
    statCpuBar.style.width = `${Math.min(100, stats.cpu.percent)}%`;

    statRamValue.textContent =
      `${Math.round(stats.memory.percent)}% (${(stats.memory.used_mb / 1024).toFixed(1)}/${(stats.memory.total_mb / 1024).toFixed(1)} GB)`;
    statRamBar.style.width = `${Math.min(100, stats.memory.percent)}%`;

    if (stats.gpu) {
      statGpuRow.hidden = false;
      statGpuLabel.textContent = stats.gpu.name;
      const { utilization_percent: util, memory_percent: memPct } = stats.gpu;
      const parts = [];
      if (util != null) parts.push(`${Math.round(util)}% util`);
      if (memPct != null) parts.push(`${Math.round(memPct)}% VRAM`);
      statGpuValue.textContent = parts.length ? parts.join(" · ") : "active";
      statGpuBar.style.width = `${Math.min(100, util ?? memPct ?? 0)}%`;
    } else {
      statGpuRow.hidden = true;
    }
  } catch {
    // best-effort — a missed poll isn't worth bothering the user about
  }
}

/* ---------- chat ---------- */
function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

newChatBtn.addEventListener("click", async () => {
  if (conversationId) {
    try {
      await fetch(`/api/conversations/${conversationId}/clear`, { method: "POST" });
    } catch {
      // best-effort — starting a new conversation client-side still works either way
    }
  }
  conversationId = null;
  chat.querySelectorAll(".message").forEach((n) => n.remove());
  chat.appendChild(emptyState);
  emptyState.style.display = "";
});

function addUserMessage(text) {
  emptyState.style.display = "none";
  const node = messageTemplate.content.cloneNode(true);
  const article = node.querySelector(".message");
  article.classList.add("message-user");
  article.querySelector(".message-bubble").textContent = text;
  chat.appendChild(article);
  scrollToBottom();
}

function addPendingAssistantMessage() {
  emptyState.style.display = "none";
  const node = messageTemplate.content.cloneNode(true);
  const article = node.querySelector(".message");
  article.classList.add("message-assistant", "pending");
  const bubble = article.querySelector(".message-bubble");
  bubble.innerHTML = '<span class="typing-dots">Thinking</span>';
  chat.appendChild(article);
  scrollToBottom();
  return article;
}

function groundednessClass(label) {
  return label === "grounded" ? "good" : "critical";
}

function renderAnswer(pendingNode, result) {
  const node = answerTemplate.content.cloneNode(true);
  const article = node.querySelector(".message-assistant");

  article.querySelector(".answer-text").textContent = result.answer;

  const g = result.groundedness || { overall_score: 0, label: "unknown", sentences: [] };
  const gClass = groundednessClass(g.label);
  const badge = article.querySelector(".badge");
  badge.classList.add(gClass);
  article.querySelector(".badge-icon").textContent = gClass === "good" ? "✓" : "⚠";
  article.querySelector(".badge-label").textContent =
    g.label === "grounded" ? "Grounded" : g.label === "unknown" ? "No context" : "Possibly hallucinated";
  article.querySelector(".badge-score").textContent = `${Math.round(g.overall_score * 100)}%`;

  const sentenceList = article.querySelector(".sentence-list");
  const sentenceBlock = article.querySelector(".sentence-breakdown");
  if (g.sentences.length === 0) {
    sentenceBlock.style.display = "none";
  } else {
    for (const s of g.sentences) {
      const cls = s.entailment >= 0.5 ? "good" : "critical";
      const li = document.createElement("li");
      li.className = "sentence-row";
      if (s.source_index != null) {
        li.classList.add("clickable");
        li.dataset.sourceIndex = s.source_index;
        li.title = "Click to jump to the source that backs this sentence";
      }
      li.innerHTML = `
        <span class="sentence-text">${s.sentence}</span>
        <span class="meter ${cls}">
          <span class="meter-fill ${cls}" style="width:${Math.round(s.entailment * 100)}%"></span>
        </span>
      `;
      sentenceList.appendChild(li);
    }
  }

  const sourceList = article.querySelector(".source-list");
  const sourcesBlock = article.querySelector(".sources-block");
  if (!result.sources || result.sources.length === 0) {
    sourcesBlock.style.display = "none";
  } else {
    result.sources.forEach((hit, index) => {
      const li = document.createElement("li");
      li.className = "source-item";
      li.dataset.sourceIndex = index;
      li.innerHTML = `
        <div class="source-item-head">
          <span>${basename(hit.source)}</span>
          <span>distance ${hit.distance.toFixed(3)}</span>
        </div>
        <div class="source-item-text">${hit.text}</div>
      `;
      sourceList.appendChild(li);
    });
  }

  // Citation highlighting: click a sentence to jump to (and highlight) the
  // exact retrieved chunk the groundedness scorer used to back it.
  sentenceList.addEventListener("click", (e) => {
    const row = e.target.closest(".sentence-row.clickable");
    if (!row) return;
    sourcesBlock.open = true;
    const target = sourceList.querySelector(`[data-source-index="${row.dataset.sourceIndex}"]`);
    if (!target) return;
    sourceList.querySelectorAll(".source-item.highlighted").forEach((n) => n.classList.remove("highlighted"));
    target.classList.add("highlighted");
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });

  pendingNode.replaceWith(article);
  scrollToBottom();
}

async function askQuestion(question) {
  addUserMessage(question);
  const pending = addPendingAssistantMessage();

  sendBtn.disabled = true;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topkInput.value),
        label: askLabelSelect.value || null,
        conversation_id: conversationId,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const result = await res.json();
    conversationId = result.conversation_id || conversationId;
    renderAnswer(pending, result);
  } catch (err) {
    pending.classList.remove("pending");
    pending.querySelector(".message-bubble").textContent =
      "Something went wrong answering that question. Check that Ollama is running and try again.";
    console.error(err);
  } finally {
    sendBtn.disabled = false;
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  questionInput.style.height = "auto";
  askQuestion(question);
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
});

/* ---------- init ---------- */
checkHealth();
refreshLabels();
refreshDevices();
refreshPerformanceMode();
refreshModels();
refreshStats();
setInterval(checkHealth, 15000);
setInterval(refreshStats, 2500);
