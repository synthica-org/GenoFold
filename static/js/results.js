(function () {
  "use strict";

  const jobId = document.body.dataset.jobId;
  const canvas = document.getElementById("structure-canvas");

  let scene, camera, renderer, pointsGroup, lineGroup;
  let autoSpin = true;
  let dragState = null;
  let rotY = 0.4, rotX = -0.2;

  const TEAL  = 0x4fb8a6;
  const AMBER = 0xe8a33d;
  const CORAL = 0xe2675b;

  function lerpColor(a, b, t) {
    const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
    const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
    return ((Math.round(ar + (br - ar) * t)) << 16) |
           ((Math.round(ag + (bg - ag) * t)) << 8) |
            Math.round(ab + (bb - ab) * t);
  }

  function confidenceColor(c) {
    if (c < 0.5) return lerpColor(CORAL, AMBER, c / 0.5);
    return lerpColor(AMBER, TEAL, (c - 0.5) / 0.5);
  }

  function colorHex(c) {
    return "#" + c.toString(16).padStart(6, "0");
  }

  function initScene() {
    scene = new THREE.Scene();
    const wrap = canvas.parentElement;
    const w = wrap.clientWidth || 600;
    const h = wrap.clientHeight || 500;

    camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    camera.position.set(0, 0, 2.8);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);

    pointsGroup = new THREE.Group();
    lineGroup   = new THREE.Group();
    scene.add(lineGroup);
    scene.add(pointsGroup);

    // subtle ambient light so spheres don't look totally flat
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);

    canvas.addEventListener("mousedown", (e) => {
      dragState = { x: e.clientX, y: e.clientY, rotY, rotX };
      autoSpin = false;
    });
    window.addEventListener("mouseup",   () => { dragState = null; });
    window.addEventListener("mousemove", (e) => {
      if (!dragState) return;
      rotY = dragState.rotY + (e.clientX - dragState.x) * 0.006;
      rotX = dragState.rotX + (e.clientY - dragState.y) * 0.006;
    });
    canvas.addEventListener("touchstart", (e) => {
      const t = e.touches[0];
      dragState = { x: t.clientX, y: t.clientY, rotY, rotX };
      autoSpin = false;
    }, { passive: true });
    window.addEventListener("touchend",  () => { dragState = null; });
    window.addEventListener("touchmove", (e) => {
      if (!dragState) return;
      const t = e.touches[0];
      rotY = dragState.rotY + (t.clientX - dragState.x) * 0.006;
      rotX = dragState.rotX + (t.clientY - dragState.y) * 0.006;
    }, { passive: true });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      camera.position.z = Math.max(1.0, Math.min(7, camera.position.z + e.deltaY * 0.0015));
    }, { passive: false });

    window.addEventListener("resize", onResize);
    animate();
  }

  function onResize() {
    const wrap = canvas.parentElement;
    const w = wrap.clientWidth || 600;
    const h = wrap.clientHeight || 500;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function buildStructure(bins) {
    while (pointsGroup.children.length) pointsGroup.remove(pointsGroup.children[0]);
    while (lineGroup.children.length)   lineGroup.remove(lineGroup.children[0]);

    if (!bins || bins.length === 0) return;

    // centre the structure
    let cx = 0, cy = 0, cz = 0;
    bins.forEach((b) => { cx += b.x; cy += b.y; cz += b.z; });
    cx /= bins.length; cy /= bins.length; cz /= bins.length;

    // scale so it fits nicely in view
    let maxR = 0;
    bins.forEach((b) => {
      const r = Math.sqrt((b.x - cx) ** 2 + (b.y - cy) ** 2 + (b.z - cz) ** 2);
      if (r > maxR) maxR = r;
    });
    const scale = maxR > 0 ? 0.9 / maxR : 1;

    const positions = bins.map((b) => new THREE.Vector3(
      (b.x - cx) * scale, (b.y - cy) * scale, (b.z - cz) * scale
    ));

    // backbone
    const lineMat = new THREE.LineBasicMaterial({ color: 0x2a3340, transparent: true, opacity: 0.6 });
    const lineGeo = new THREE.BufferGeometry().setFromPoints(positions);
    lineGroup.add(new THREE.Line(lineGeo, lineMat));

    // one sphere geometry, many mesh instances
    const sphereGeo = new THREE.SphereGeometry(0.022, 14, 14);
    bins.forEach((b, i) => {
      const mat = new THREE.MeshLambertMaterial({ color: confidenceColor(b.confidence) });
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.position.copy(positions[i]);
      // slightly scale by influence
      const inf = Math.min(b.influence || 0.5, 1);
      const s = 0.7 + inf * 0.7;
      mesh.scale.set(s, s, s);
      pointsGroup.add(mesh);
    });
  }

  function animate() {
    requestAnimationFrame(animate);
    if (autoSpin) rotY += 0.003;
    pointsGroup.rotation.set(rotX, rotY, 0);
    lineGroup.rotation.set(rotX, rotY, 0);
    renderer.render(scene, camera);
  }

  document.getElementById("btn-spin").addEventListener("click", () => {
    autoSpin = !autoSpin;
    document.getElementById("btn-spin").textContent = autoSpin ? "\u23F5" : "\u23F8";
  });

  document.getElementById("btn-reset-view").addEventListener("click", () => {
    rotY = 0.4; rotX = -0.2;
    camera.position.set(0, 0, 2.8);
  });

  function renderRegions(regions) {
    const el = document.getElementById("regions-list");
    el.innerHTML = "";
    if (!regions || !regions.length) {
      el.innerHTML = '<p style="font-size:12px;color:var(--paper-faint)">No regions detected.</p>';
      return;
    }
    regions.forEach((r) => {
      const pct  = Math.round(r.mean_confidence * 100);
      const col  = colorHex(confidenceColor(r.mean_confidence));
      const row  = document.createElement("div");
      row.className = "region-row";
      row.innerHTML = `
        <span class="region-label">bins ${r.start_bin}&ndash;${r.end_bin}</span>
        <div class="region-bar-track">
          <div class="region-bar-fill" style="width:${pct}%;background:${col}"></div>
        </div>
        <span class="region-conf">${pct}%</span>`;
      el.appendChild(row);
    });
  }

  function renderFeatures(features) {
    const el = document.getElementById("feature-list");
    el.innerHTML = "";
    if (!features) return;
    const vals = Object.values(features);
    const max  = Math.max(...vals, 1e-12);
    Object.entries(features).forEach(([name, val]) => {
      const pct = Math.max(4, Math.round((val / max) * 100));
      const row = document.createElement("div");
      row.className = "feat-row";
      row.innerHTML = `
        <span class="feat-label">${name}</span>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:${pct}%"></div>
        </div>`;
      el.appendChild(row);
    });
  }

  function renderLoci(indices) {
    const el = document.getElementById("loci-chips");
    el.innerHTML = "";
    (indices || []).forEach((idx) => {
      const chip = document.createElement("span");
      chip.className = "loci-chip";
      chip.textContent = `bin\u00a0${idx}`;
      el.appendChild(chip);
    });
  }

  function figUrl(path) {
    return `/outputs/${jobId}/${path}`;
  }

  async function loadResults() {
    try {
      const res  = await fetch(`/api/results/${jobId}`);
      const data = await res.json();
      if (!data.ok) {
        console.error("Results not ready:", data.error);
        return;
      }

      // header
      document.getElementById("ds-name").textContent  = data.dataset_name || "Upload";
      document.getElementById("ds-chrom").textContent = data.chrom;
      document.getElementById("ds-res").textContent   = `${(data.resolution / 1000).toFixed(0)} kb`;

      // metrics
      document.getElementById("metric-bins").textContent  = data.num_bins;
      document.getElementById("metric-edges").textContent = data.num_edges.toLocaleString();
      document.getElementById("metric-conf").textContent  = data.mean_confidence.toFixed(3);
      document.getElementById("metric-unc").innerHTML     =
        `${data.mean_uncertainty.toFixed(3)}<span class="unit">radius</span>`;

      // 3D
      buildStructure(data.bins);

      // sidebar panels
      renderRegions(data.regions);
      renderFeatures(data.feature_importance);
      renderLoci(data.top_influential_loci);

      // downloads
      document.getElementById("dl-pdb").href = figUrl(data.downloads.pdb);
      document.getElementById("dl-npy").href = figUrl(data.downloads.npy);
      document.getElementById("dl-tsv").href = figUrl(data.downloads.tsv);
      document.getElementById("btn-export-pdb").addEventListener("click", () => {
        window.location.href = figUrl(data.downloads.pdb);
      });

      // explainability iframes
      const figs = data.figures;
      function loadFig(frameId, openId, path) {
        const url = figUrl(path);
        document.getElementById(frameId).src = url;
        document.getElementById(openId).href = url;
      }
      loadFig("frame-feat",   "open-feat",   figs.feature_importance_html);
      loadFig("frame-attn",   "open-attn",   figs.attention_heatmap_html);
      loadFig("frame-loci",   "open-loci",   figs.influential_loci_html);
      loadFig("frame-struct", "open-struct", figs.structure_confidence_html);

    } catch (err) {
      console.error("Failed to load results:", err);
    }
  }

  initScene();
  loadResults();
})();
