/* JIT Episodic Memory — project page interactions */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const J = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p); return r.json(); });

  const INK = "#141726", MUT = "#5b6172", GRID = "#e9ecf4";
  const A = "#5457e6", A2 = "#12b3a6", WARM = "#f0873a", GRAY = "#9aa0b4";

  /* ---------- cascade detail panel ---------- */
  const STAGES = [
    { col: "#5457e6", title: "Level 1 — Semantic retrieval",
      body: "Embed the text query with CLIP and search the FAISS index of stored keyframe embeddings; keep the k = 100 most similar frames.",
      chips: ["CLIP ViT-B/32", "FAISS flat · inner product", "k = 100 (20% of 500)"],
      points: ["Sub-millisecond nearest-neighbour search, O(1) insertion as the robot explores.",
               "Bounds all downstream detection to a fixed k frames, so query cost is independent of how large the memory grows."] },
    { col: "#12b3a6", title: "Level 2 — Detect · project · cluster",
      body: "Run an open-vocabulary detector on each retrieved frame, back-project each detection to 3D with depth, and merge consistent cross-view hits with DBSCAN. Clusters are ranked by cumulative detection confidence.",
      chips: ["OWL-ViT · τ = 0.1", "depth 30th-pct · 30×30", "DBSCAN · ε = 1 m"],
      points: ["A spatial denoiser: a real object forms a tight, high-confidence cluster; spurious detections scatter into low-ranked singletons.",
               "The only stage that runs the expensive detector — and only on the k retrieved frames."] },
    { col: "#f0873a", title: "Level 3 — Verification (optional)",
      body: "Re-detect the query in the top-5 cluster frames and refine their 3D centroids. A capture-density rule gates it automatically.",
      chips: ["top-5 clusters", "dense captures only", "+2.2 pp Loc@1 m (ScanNet)"],
      points: ["Helps dense video (ScanNet) but hurts wide-baseline scans (Replica), so the base system is L1+L2 and L3 is reported as an ablation.",
               "Enabled when median frame spacing < 0.5 m; disabled on HM3D."] },
  ];
  function showStage(i) {
    const s = STAGES[i], d = $("#detail");
    d.style.setProperty("--stagecol", s.col);
    d.innerHTML = `<h4>${s.title}</h4><p style="color:var(--muted);margin:0 0 4px">${s.body}</p>
      <div style="margin:14px 0 2px">${s.chips.map((c) => `<span class="chip">${c}</span>`).join("")}</div>
      <ul>${s.points.map((p) => `<li>${p}</li>`).join("")}</ul>`;
    $$("#stages .stage").forEach((el, k) => el.classList.toggle("active", k === i));
  }
  $$("#stages .stage").forEach((el) => {
    const i = +el.dataset.stage;
    el.addEventListener("mouseenter", () => showStage(i));
    el.addEventListener("click", () => showStage(i));
  });
  showStage(0);

  /* ---------- charts ---------- */
  if (window.Chart) {
    Chart.defaults.font.family = "system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
    Chart.defaults.font.size = 13;
    Chart.defaults.color = MUT;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
  }
  const axis = (title, opts = {}) => Object.assign(
    { grid: { color: GRID, drawBorder: false }, title: { display: !!title, text: title, color: MUT } }, opts);
  const line = (c, w = 2.5) => ({ borderColor: c, backgroundColor: c, borderWidth: w, tension: .3,
    pointRadius: 3, pointHoverRadius: 5, pointBackgroundColor: c });

  // Budget curve
  J("static/data/budget.json").then((d) => {
    new Chart($("#chartBudget"), {
      type: "line",
      data: { labels: d.k, datasets: [
        Object.assign({ label: "JIT", data: d.arms.jit.loc_1m }, line(A, 3)),
        Object.assign({ label: "Random frames", data: d.arms.random.loc_1m, borderDash: [6, 5] }, line(GRAY, 2)),
      ] },
      options: { responsive: true, maintainAspectRatio: false,
        scales: { x: axis("detector budget k (frames)"), y: axis("Loc@1 m (%)", { suggestedMin: 50, suggestedMax: 85 }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y}%` } } } },
    });
  }).catch(() => {});

  // Scalability latency
  J("static/data/scalability.json").then((d) => {
    const S = d.series;
    new Chart($("#chartScale"), {
      type: "line",
      data: { labels: d.frames, datasets: [
        Object.assign({ label: "JIT (k=100)", data: S.jit.latency_s }, line(A, 3)),
        Object.assign({ label: "Brute force (100)", data: S.bf100.latency_s }, line(A2, 2)),
        Object.assign({ label: "Brute force (all frames)", data: S.bf_all.latency_s }, line(WARM, 2)),
      ] },
      options: { responsive: true, maintainAspectRatio: false,
        scales: { x: axis("frames stored in memory"), y: axis("query latency (s)", { beginAtZero: true }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y}s` } } } },
    });
  }).catch(() => {});

  // Build cost (log horizontal bar)
  J("static/data/results_scannet.json").then((d) => {
    const m = d.methods.filter((x) => x.name !== "JIT (+L3)"); // one JIT row
    m.forEach((x) => { if (x.name === "JIT (L1+L2)") x.name = "JIT"; });
    new Chart($("#chartBuild"), {
      type: "bar",
      data: { labels: m.map((x) => x.name), datasets: [{
        label: "build time (s)", data: m.map((x) => x.build_s),
        backgroundColor: m.map((x) => x.name === "JIT" ? A : "#c7ccdd"),
        borderRadius: 6, barThickness: 22 }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
        scales: { x: axis("seconds (log scale)", { type: "logarithmic" }), y: axis("") },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: (c) => `${c.parsed.x}s to build` } } } },
    });
  }).catch(() => {});

  // Cross-dataset grouped bars
  J("static/data/cross_dataset.json").then((d) => {
    const pick = ["JIT (L1+L2)", "Random-100+DBSCAN", "ConceptGraphs", "GOAT"];
    const cols = { "JIT (L1+L2)": A, "Random-100+DBSCAN": GRAY, "ConceptGraphs": WARM, "GOAT": A2 };
    const idx = pick.map((p) => d.columns.indexOf(p));
    new Chart($("#chartCross"), {
      type: "bar",
      data: { labels: d.rows.map((r) => `${r.dataset} (${r.scenes})`),
        datasets: pick.map((p, j) => ({ label: p.replace(" (L1+L2)", ""),
          data: d.rows.map((r) => r.vals[idx[j]]), backgroundColor: cols[p], borderRadius: 4 })) },
      options: { responsive: true, maintainAspectRatio: false,
        scales: { x: axis(""), y: axis("Loc@1 m (%)", { beginAtZero: true, suggestedMax: 90 }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y == null ? "n/a" : c.parsed.y + "%"}` } } } },
    });
  }).catch(() => {});

  // Crossover: cumulative wall-clock cost vs #queries (JIT vs heaviest eager map)
  J("static/data/crossover.json").then((d) => {
    const N = 1000, xs = [];
    for (let n = 0; n <= N; n += 25) xs.push(n);
    const jit = xs.map((n) => +(d.jit.build_s + n * d.jit.query_s).toFixed(1));
    const cg = d.eager.find((e) => e.name === "ConceptGraphs");
    const eg = xs.map((n) => +(cg.build_s + n * cg.query_s).toFixed(1));
    new Chart($("#chartCrossover"), {
      type: "line",
      data: { labels: xs, datasets: [
        Object.assign({ label: "JIT (build 3.1 s + 2.5 s/query)", data: jit, fill: false }, line(A, 3)),
        Object.assign({ label: "ConceptGraphs (build 2,160 s + ~0 /query)", data: eg, fill: false }, line(WARM, 2.5)),
        { label: `crossover ≈ ${cg.crossover} queries`, data: [{ x: cg.crossover, y: +(d.jit.build_s + cg.crossover * d.jit.query_s).toFixed(0) }],
          type: "scatter", pointRadius: 7, pointHoverRadius: 8, pointStyle: "rectRot",
          backgroundColor: "#141726", borderColor: "#fff", borderWidth: 2 },
      ] },
      options: { responsive: true, maintainAspectRatio: false, parsing: true,
        scales: { x: axis("queries per scene", { type: "linear", max: N }), y: axis("cumulative wall-clock time (s)", { beginAtZero: true }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}` } } } },
    });
  }).catch(() => {});

  /* ---------- interactive query demo ---------- */
  J("static/data/query_demo.json").then((d) => {
    const sel = $("#q"), res = $("#qresult"), fr = $("#qframes");
    d.queries.forEach((q, i) => {
      const o = document.createElement("option"); o.value = i; o.textContent = q.query; sel.appendChild(o);
    });
    function render(i) {
      const q = d.queries[i];
      fr.innerHTML = q.frames.map((f) =>
        `<figure class="hit"><span class="tag">det ${f.score}</span><img src="${f.src}" alt="detected frame" loading="lazy"></figure>`).join("");
      const r = q.result;
      res.innerHTML = (r.detected !== false)
        ? `<div class="k">JIT localizes the ${q.query} at</div>
           <div class="v" style="font-size:18px;color:var(--accent)">(${r.x}, ${r.y}, ${r.z}) m</div>
           <div style="margin-top:12px"><span class="k">error to ground truth</span> <span class="v">${r.error_m} m</span></div>
           <div><span class="k">detected in</span> <span class="v">${r.n_detected} keyframes</span></div>
           <div><span class="k">query latency</span> <span class="v">${r.latency_ms} ms</span></div>`
        : `<div class="k">No confident detection — JIT reports the object as absent rather than guessing.</div>`;
    }
    sel.addEventListener("change", (e) => render(+e.target.value));
    render(0);
  }).catch(() => {
    const box = $("#demo");
    if (box) box.innerHTML = '<p class="lead" style="margin:0">Interactive demo assets are generated by <code>docs/make_footage.py</code>.</p>';
  });

  /* ---------- bibtex copy ---------- */
  const cb = $("#copyBib");
  if (cb) cb.addEventListener("click", () => {
    navigator.clipboard.writeText($("#bibblock").textContent.trim()).then(() => {
      cb.textContent = "Copied ✓"; setTimeout(() => (cb.textContent = "Copy"), 1600);
    });
  });

  /* ---------- graceful video fallback ---------- */
  $$("[data-vid] video").forEach((v) => {
    v.addEventListener("error", () => {
      const w = v.closest("[data-vid]");
      if (w) w.innerHTML = '<div class="videofallback">Footage renders via <code>docs/make_footage.py</code>.</div>';
    });
  });
})();
