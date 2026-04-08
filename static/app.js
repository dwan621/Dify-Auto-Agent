let currentSceneKey = "generic";
let currentWorkflowSpec = null;
let currentRequirement = null;
let currentSceneLabel = "暂无";
let modelReady = false;
let uiPhase = "idle";
let healthPollId = null;

const $ = (id) => document.getElementById(id);

function makeTraceId() {
  return `web_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
}

function appendMessage(text, role = "system") {
  const chatBox = $("chatBox");
  const div = document.createElement("div");
  div.className = `message message--${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function setButtonsDisabled(disabled) {
  $("analyzeBtn").disabled = disabled;
  $("generateBtn").disabled = disabled;
  $("validateBtn").disabled = disabled;
  $("copyYamlBtn").disabled = disabled;
  const copyTop = $("copyYamlTopBtn");
  if (copyTop) {
    copyTop.disabled = disabled;
  }
}

function setPhase(phase) {
  uiPhase = phase;
  const status = $("uiStatus");
  const map = {
    idle: { text: "未分析", tone: "" },
    health_blocked: { text: "模型未就绪", tone: "warn" },
    analyzing: { text: "分析中", tone: "info" },
    analyzed: { text: "可编辑", tone: "ok" },
    validating: { text: "校验中", tone: "info" },
    invalid: { text: "待修复", tone: "warn" },
    generating: { text: "生成中", tone: "info" },
    generated: { text: "已生成", tone: "ok" },
  };
  const next = map[phase] || { text: phase, tone: "" };
  if (status) {
    status.textContent = next.text;
    if (next.tone) {
      status.setAttribute("data-tone", next.tone);
    } else {
      status.removeAttribute("data-tone");
    }
  }
  syncControls();
  syncStepper();
}

function setResultMessage(text) {
  const el = $("resultMessage");
  if (el) {
    el.textContent = text;
  }
}

function syncStepper() {
  const stepper = $("stepper");
  if (!stepper) {
    return;
  }
  const items = stepper.querySelectorAll(".stepper__item");
  const phaseToStep = {
    idle: 1,
    health_blocked: 1,
    analyzing: 2,
    analyzed: 3,
    validating: 4,
    invalid: 4,
    generating: 5,
    generated: 5,
  };
  const activeStep = phaseToStep[uiPhase] || 1;
  items.forEach((el) => {
    const s = Number(el.getAttribute("data-step") || "0");
    el.setAttribute("data-active", s === activeStep ? "true" : "false");
  });
}

function syncControls() {
  const analyzing = uiPhase === "analyzing";
  const generating = uiPhase === "generating";
  const healthBlocked = !modelReady;

  $("analyzeBtn").disabled = healthBlocked || analyzing || generating;
  $("generateBtn").disabled = healthBlocked || analyzing || generating || !currentWorkflowSpec;
  $("validateBtn").disabled = healthBlocked || analyzing || generating || !currentWorkflowSpec;

  const yamlText = $("yamlPreview").textContent || "";
  const copyDisabled = healthBlocked || analyzing || generating || !yamlText.trim();
  $("copyYamlBtn").disabled = copyDisabled;
  const copyTop = $("copyYamlTopBtn");
  if (copyTop) {
    copyTop.disabled = copyDisabled;
  }
}

function renderHealth(data) {
  const warn = $("healthWarn");
  const providerStatus = data.provider_status || {};
  const providerLines = Object.keys(providerStatus).map((name) => {
    const info = providerStatus[name] || {};
    const isReady = typeof info.ready === "boolean" ? info.ready : undefined;
    const isConfigured = typeof info.configured === "boolean" ? info.configured : undefined;
    const mark = isReady === true || isConfigured === true ? "OK" : isReady === false || isConfigured === false ? "FAIL" : "UNKNOWN";
    const detail = info.model || info.message || info.base_url || "";
    return `${name}: ${mark}${detail ? ` (${detail})` : ""}`;
  });

  const isModelReady =
    data.model_ready === true ||
    data.model_ready === "true" ||
    data.model_ready === 1 ||
    data.model_ready === "1" ||
    (!!data.active_provider && !!data.active_model);

  if (!isModelReady) {
    modelReady = false;
    const tips = [
      "当前无可用模型，分析/生成已禁用。",
      "请在 .env 配置 DEEPSEEK_API_KEY 或 FALLBACK_API_KEY/BASE_URL/MODEL。",
      data.deepseek_status ? `当前状态：${data.deepseek_status}` : "",
      ...providerLines,
    ];
    if (data.broken_proxy_detected) {
      tips.push("检测到异常代理设置，请检查 HTTP_PROXY/HTTPS_PROXY。");
    }
    if (data.generated_dir_writable === false) {
      tips.push("生成目录不可写，请检查 generated/ 权限。");
    }
    warn.style.display = "block";
    warn.textContent = tips.filter(Boolean).join("\n");
    setPhase("health_blocked");
    startHealthPolling();
    return;
  }

  modelReady = true;

  warn.style.display = "none";
  stopHealthPolling();
  if (uiPhase === "health_blocked") {
    setPhase(currentWorkflowSpec ? "analyzed" : "idle");
  }
  syncControls();
}

function stopHealthPolling() {
  if (healthPollId) {
    clearInterval(healthPollId);
    healthPollId = null;
  }
}

function startHealthPolling() {
  if (healthPollId) {
    return;
  }
  healthPollId = setInterval(() => {
    loadHealth();
  }, 15000);
}

function renderCandidateList(candidates = [], selectedId = "") {
  const list = $("candidateList");
  list.innerHTML = "";

  if (!candidates.length) {
    const li = document.createElement("li");
    li.className = "list__item";
    li.textContent = "暂无候选评分。先完成“分析需求”。";
    list.appendChild(li);
    return;
  }

  candidates.forEach((cand) => {
    const li = document.createElement("li");
    li.className = "list__item";
    const ratio = typeof cand.non_llm_node_ratio === "number" ? cand.non_llm_node_ratio : 0;
    const head = document.createElement("div");
    head.style.display = "flex";
    head.style.alignItems = "center";
    head.style.justifyContent = "space-between";
    head.style.gap = "10px";

    const left = document.createElement("div");
    left.style.display = "flex";
    left.style.alignItems = "center";
    left.style.gap = "8px";

    const id = document.createElement("span");
    id.textContent = cand.id;
    id.className = "mono";
    left.appendChild(id);

    if (cand.id === selectedId) {
      const tag = document.createElement("span");
      tag.className = "tag tag--selected";
      tag.textContent = "已选";
      left.appendChild(tag);
    }

    const right = document.createElement("div");
    right.className = "mono muted";
    right.textContent = `score=${cand.score} | non-llm=${ratio}`;

    head.appendChild(left);
    head.appendChild(right);
    li.appendChild(head);
    list.appendChild(li);
  });
}

function renderWarnings(warnings = []) {
  const list = $("warningList");
  list.innerHTML = "";

  if (!warnings.length) {
    const li = document.createElement("li");
    li.className = "list__item";
    li.textContent = "无告警。建议生成前再点一次“仅校验 Spec”。";
    list.appendChild(li);
    return;
  }

  warnings.forEach((w) => {
    const li = document.createElement("li");
    li.className = "list__item";
    li.textContent = w;
    list.appendChild(li);
  });
}

function setInlineError(container, message) {
  if (!container) {
    return;
  }
  const existing = container.querySelector(".field-error");
  if (!message) {
    if (existing) {
      existing.remove();
    }
    return;
  }
  const el = existing || document.createElement("div");
  el.className = "field-error";
  el.textContent = message;
  if (!existing) {
    container.appendChild(el);
  }
}

function renderNodeEditor(spec) {
  const nodeList = $("nodeList");
  nodeList.innerHTML = "";

  if (!spec || !Array.isArray(spec.steps) || !spec.steps.length) {
    const li = document.createElement("li");
    li.className = "list__item";
    li.textContent = "暂无节点。先完成“分析需求”，或加载模板。";
    nodeList.appendChild(li);
    return;
  }

  spec.steps.forEach((step) => {
    const li = document.createElement("li");
    li.className = "node-editor-item";

    const details = document.createElement("details");
    details.open = false;

    const summary = document.createElement("summary");

    const sumWrap = document.createElement("div");
    sumWrap.className = "node-summary";

    const typeTag = document.createElement("span");
    typeTag.className = "tag";
    typeTag.textContent = step.type || "unknown";
    sumWrap.appendChild(typeTag);

    const titleSpan = document.createElement("span");
    titleSpan.className = "node-summary__title";
    titleSpan.textContent = step.title || step.id || "node";
    sumWrap.appendChild(titleSpan);

    const idSpan = document.createElement("span");
    idSpan.className = "mono muted";
    idSpan.textContent = step.id ? `#${step.id}` : "";

    summary.appendChild(sumWrap);
    summary.appendChild(idSpan);
    details.appendChild(summary);

    const fields = document.createElement("div");
    fields.className = "node-fields";

    const titleInput = document.createElement("input");
    titleInput.type = "text";
    titleInput.value = step.title || step.id || "";
    titleInput.placeholder = "节点标题";
    titleInput.className = "input";
    titleInput.addEventListener("input", () => {
      step.title = titleInput.value.trim() || step.id || "";
      titleSpan.textContent = step.title;
    });
    fields.appendChild(titleInput);

    if (step.prompt !== undefined) {
      const promptArea = document.createElement("textarea");
      promptArea.value = step.prompt || "";
      promptArea.placeholder = "节点提示词";
      promptArea.className = "textarea";
      promptArea.addEventListener("change", () => {
        step.prompt = promptArea.value;
      });
      fields.appendChild(promptArea);
    }

    if (step.config && typeof step.config === "object") {
      const configArea = document.createElement("textarea");
      configArea.value = JSON.stringify(step.config, null, 2);
      configArea.placeholder = "节点配置 JSON";
      configArea.className = "textarea mono";
      configArea.addEventListener("change", () => {
        try {
          const parsed = JSON.parse(configArea.value || "{}");
          step.config = parsed;
          setInlineError(fields, "");
        } catch (err) {
          setInlineError(fields, "节点配置 JSON 格式错误，未保存。请修正后再离开输入框。");
        }
      });
      fields.appendChild(configArea);
    }

    details.appendChild(fields);

    li.appendChild(details);
    nodeList.appendChild(li);
  });
}

function updatePreview(spec, sceneLabel) {
  $("sceneLabel").textContent = sceneLabel || "暂无";
  renderNodeEditor(spec);
  syncControls();
}

function getQuickTemplateText(key) {
  const map = {
    translate: "创建一个多语言翻译工作流：识别语种、翻译、质量校验、格式化输出。",
    summary: "创建一个文档总结工作流：抽取要点、风险、行动项，并给出结构化摘要。",
    complex: "创建一个复杂工作流：先做意图理解，再分支执行检索、工具调用和代码处理，最后统一格式化输出。",
  };
  return map[key] || "";
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    const isJson = contentType.includes("application/json");
    const data = isJson ? await res.json() : { model_ready: false, raw: await res.text() };
    if (!res.ok && isJson && !data.model_ready) {
      renderHealth(data);
      return;
    }
    if (!res.ok && !isJson) {
      throw new Error(`health ${res.status}`);
    }
    renderHealth(data);
  } catch (error) {
    const warn = $("healthWarn");
    warn.style.display = "block";
    warn.replaceChildren();
    const msg = document.createElement("div");
    msg.textContent = `健康检查失败：${error.message}`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "button button--secondary";
    btn.textContent = "重试";
    btn.addEventListener("click", () => loadHealth());
    warn.appendChild(msg);
    warn.appendChild(document.createElement("div"));
    warn.appendChild(btn);
    modelReady = false;
    setPhase("health_blocked");
    startHealthPolling();
  }
}

function applyAnalyzeResult(data) {
  currentSceneKey = data.scene_key || "generic";
  currentWorkflowSpec = data.workflow_spec || null;
  currentRequirement = data.requirement || null;
  currentSceneLabel = data.scene_label || "通用工作流";

  updatePreview(currentWorkflowSpec, currentSceneLabel);
  renderCandidateList(data.candidates_brief || [], data.selected_candidate_id || "");
  renderWarnings(data.validation_warnings || []);

  if (data.app_name_suggestion) {
    $("appName").value = data.app_name_suggestion;
  }
  if (data.description_suggestion) {
    $("appDesc").value = data.description_suggestion;
  }

  const planningText = [
    `mode=${data.planning_mode || "unknown"}`,
    `selected=${data.selected_candidate_id || "n/a"}`,
    `repair_rounds=${data.repair_rounds ?? 0}`,
    data.trace_id ? `trace_id=${data.trace_id}` : "",
  ].filter(Boolean).join(" | ");
  $("planningMeta").textContent = planningText;

  const artifactId = data.artifact_id || "";
  if (artifactId) {
    $("artifactMeta").innerHTML = `artifact: <a href="/artifact/${artifactId}" target="_blank">${artifactId}</a>`;
  }

  if (data.followup_text) {
    appendMessage(data.followup_text, "system");
  }

  setPhase("analyzed");
}

async function analyzeRequirement() {
  const userInput = $("userInput").value.trim();
  if (!userInput || userInput.length < 4) {
    setResultMessage("请提供更完整的需求描述（建议 2～5 句话）。");
    setPhase("idle");
    return;
  }

  const traceId = makeTraceId();
  setPhase("analyzing");
  $("analyzeBtn").textContent = "分析中...";
  setResultMessage("正在分析需求...");
  appendMessage(userInput, "user");

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Trace-Id": traceId,
      },
      body: JSON.stringify({ user_input: userInput }),
    });
    const data = await res.json();

    if (!res.ok) {
      if (res.status === 503 && data.guidance) {
        $("healthWarn").style.display = "block";
        $("healthWarn").textContent = `${data.error} ${data.guidance.join(" ")}`;
      }
      throw new Error(data.error || "分析失败");
    }

    applyAnalyzeResult(data);
    setResultMessage("分析完成，可在右侧编辑节点后生成 YAML。");
  } catch (error) {
    setResultMessage(`分析失败：${error.message}`);
    setPhase(currentWorkflowSpec ? "analyzed" : "idle");
  } finally {
    $("analyzeBtn").textContent = "分析需求";
    syncControls();
  }
}

function collectGeneratePayload() {
  const answersRaw = $("answersInput").value.trim();
  const answers = answersRaw
    ? answersRaw.split("\n").map((s) => s.trim()).filter(Boolean)
    : [];

  const maxExecRaw = $("maxExecNodes").value.trim();
  const maxExecNodes = maxExecRaw === "" ? null : Number(maxExecRaw);

  return {
    scene_key: currentSceneKey || "generic",
    scene_label: currentSceneLabel || "通用工作流",
    app_name: $("appName").value.trim(),
    description: $("appDesc").value.trim(),
    answers,
    workflow_spec: currentWorkflowSpec,
    requirement: currentRequirement,
    strict_validation: true,
    budget_level: $("budgetLevel").value,
    max_exec_nodes: Number.isFinite(maxExecNodes) ? maxExecNodes : null,
    import_mode: $("autoImport").checked ? "download_and_import" : "download_only",
    auto_import_to_dify: $("autoImport").checked,
  };
}

async function generateYaml() {
  if (!currentWorkflowSpec) {
    setResultMessage("当前没有可生成的工作流：请先分析需求或加载模板。");
    setPhase("idle");
    return;
  }

  const traceId = makeTraceId();
  setPhase("generating");
  $("generateBtn").textContent = "生成中...";
  setResultMessage("正在生成 YAML...");

  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Trace-Id": traceId,
      },
      body: JSON.stringify(collectGeneratePayload()),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "生成失败");
    }

    $("yamlPreview").textContent = data.yaml_content || "";
    setResultMessage(`${data.message}（${data.filename}）`);

    const downloadLink = $("downloadLink");
    downloadLink.href = data.download_url;
    downloadLink.style.display = "inline-block";

    if (data.workflow_spec) {
      currentWorkflowSpec = data.workflow_spec;
      updatePreview(currentWorkflowSpec, currentSceneLabel);
    }

    const importStatus = data.dify_import_status || "not_requested";
    if (importStatus === "success") {
      $("artifactMeta").textContent = `Dify 导入成功，job_id=${data.dify_import?.job_id || "n/a"}`;
    } else if (importStatus === "failed") {
      $("artifactMeta").textContent = "Dify 导入失败（YAML 已可下载），请检查 Dify 配置。";
    }

    setPhase("generated");
  } catch (error) {
    setResultMessage(`生成失败：${error.message}`);
    setPhase("invalid");
  } finally {
    $("generateBtn").textContent = "生成 YAML";
    syncControls();
  }
}

async function validateSpecOnly() {
  if (!currentWorkflowSpec) {
    setResultMessage("暂无可校验的 Spec。先完成“分析需求”。");
    return;
  }

  try {
    setPhase("validating");
    const res = await fetch("/validate_spec", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow_spec: currentWorkflowSpec }),
    });
    const data = await res.json();
    renderWarnings([...(data.errors || []), ...(data.warnings || [])]);
    setResultMessage(data.ok ? "Spec 校验通过，可继续生成。" : "Spec 校验未通过，请先处理告警后再生成。");
    setPhase(data.ok ? "analyzed" : "invalid");
  } catch (error) {
    setResultMessage(`校验失败：${error.message}`);
    setPhase(currentWorkflowSpec ? "analyzed" : "idle");
  }
}

async function loadComplexTemplate() {
  try {
    const res = await fetch("/template/complex_demo");
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "模板加载失败");
    }

    currentWorkflowSpec = data.workflow_spec;
    currentRequirement = {
      scene: "generic",
      app_name: currentWorkflowSpec.workflow_name || "复杂示例工作流",
      description: currentWorkflowSpec.description || "",
    };
    currentSceneKey = "generic";
    currentSceneLabel = currentWorkflowSpec.workflow_name || "复杂示例工作流";

    updatePreview(currentWorkflowSpec, currentSceneLabel);
    renderWarnings([...(data.validation?.warnings || [])]);
    renderCandidateList([], "");

    $("appName").value = currentWorkflowSpec.workflow_name || "";
    $("appDesc").value = currentWorkflowSpec.description || "";
    $("planningMeta").textContent = "已加载 complex_demo 模板";
    setResultMessage("复杂模板已加载，可直接生成 YAML。建议先校验 Spec。 ");
    setPhase("analyzed");
  } catch (error) {
    setResultMessage(`加载模板失败：${error.message}`);
  }
}

function setAllNodeDetailsOpen(open) {
  const nodeList = $("nodeList");
  if (!nodeList) {
    return;
  }
  nodeList.querySelectorAll("details").forEach((d) => {
    d.open = open;
  });
}

function bindEvents() {
  $("quickTemplate").addEventListener("change", (event) => {
    const value = event.target.value;
    const text = getQuickTemplateText(value);
    if (text) {
      $("userInput").value = text;
    }
  });

  $("analyzeBtn").addEventListener("click", analyzeRequirement);
  $("generateBtn").addEventListener("click", generateYaml);
  $("validateBtn").addEventListener("click", validateSpecOnly);
  $("loadComplexBtn").addEventListener("click", loadComplexTemplate);

  $("copyYamlBtn").addEventListener("click", async () => {
    const yamlText = $("yamlPreview").textContent || "";
    if (!yamlText) {
      return;
    }
    await navigator.clipboard.writeText(yamlText);
    setResultMessage("YAML 已复制到剪贴板");
  });

  const copyTop = $("copyYamlTopBtn");
  if (copyTop) {
    copyTop.addEventListener("click", async () => {
      const yamlText = $("yamlPreview").textContent || "";
      if (!yamlText) {
        return;
      }
      await navigator.clipboard.writeText(yamlText);
      setResultMessage("YAML 已复制到剪贴板");
    });
  }

  const expandAllBtn = $("expandAllBtn");
  if (expandAllBtn) {
    expandAllBtn.addEventListener("click", () => setAllNodeDetailsOpen(true));
  }
  const collapseAllBtn = $("collapseAllBtn");
  if (collapseAllBtn) {
    collapseAllBtn.addEventListener("click", () => setAllNodeDetailsOpen(false));
  }
}

bindEvents();
loadHealth();
setPhase("idle");
