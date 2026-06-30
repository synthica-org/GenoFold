(function () {
  "use strict";

  const slots = document.querySelectorAll(".slot");
  const filesBySlot = {};

  slots.forEach((slot) => {
    const key = slot.dataset.slot;
    const input = slot.querySelector('input[type="file"]');
    const dropzone = slot.querySelector(".slot-dropzone");
    const originalLabel = dropzone.textContent;

    function showFile(file) {
      filesBySlot[key] = file;
      slot.classList.add("is-filled");
      dropzone.classList.add("has-file");
      dropzone.innerHTML = "";
      const name = document.createElement("span");
      name.className = "slot-filename";
      name.textContent = file.name;
      const clear = document.createElement("span");
      clear.className = "slot-clear";
      clear.textContent = "remove";
      clear.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearFile();
      });
      dropzone.appendChild(name);
      dropzone.appendChild(clear);
    }

    function clearFile() {
      delete filesBySlot[key];
      input.value = "";
      slot.classList.remove("is-filled");
      dropzone.classList.remove("has-file");
      dropzone.textContent = originalLabel;
    }

    input.addEventListener("change", () => {
      if (input.files && input.files[0]) showFile(input.files[0]);
    });

    ["dragenter", "dragover"].forEach((evt) => {
      slot.addEventListener(evt, (e) => {
        e.preventDefault();
        slot.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      slot.addEventListener(evt, (e) => {
        e.preventDefault();
        slot.classList.remove("is-dragover");
      });
    });
    slot.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) {
        input.files = e.dataTransfer.files;
        showFile(file);
      }
    });
  });

  const form = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");
  const errorsBox = document.getElementById("errors-box");
  const loadingScreen = document.getElementById("loading-screen");
  const loadingStage = document.getElementById("loading-stage");
  const loadingMessage = document.getElementById("loading-message");
  const loadingBar = document.getElementById("loading-bar");
  const loadingPercent = document.getElementById("loading-percent");
  const loadingError = document.getElementById("loading-error");

  const STAGE_LABELS = {
    load: "Stage 0 — loading data",
    preprocess: "Stage 1 — preprocess",
    represent: "Stage 2 — represent",
    infer: "Stage 3 — infer",
    reconstruct: "Stage 4 — reconstruct",
    score: "Stage 5 — score",
    explain: "Stage 6 — explain",
    finalize: "Finalizing",
    done: "Done",
  };

  function showErrors(list) {
    errorsBox.innerHTML = list.map((e) => `<div>${e}</div>`).join("");
    errorsBox.classList.add("show");
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorsBox.classList.remove("show");

    const hicPathInput = form.querySelector('input[name="hic_path"]');
    const hasHicFile = !!filesBySlot.hic;
    const hasHicPath = hicPathInput && hicPathInput.value.trim().length > 0;
    if (!hasHicFile && !hasHicPath) {
      showErrors(["A Hi-C file or file path is required before running the pipeline."]);
      return;
    }

    submitBtn.disabled = true;
    loadingScreen.classList.add("active");
    initThread();

    const fd = new FormData(form);

    try {
      const res = await fetch("/api/submit", { method: "POST", body: fd });
      const data = await res.json();

      if (!data.ok) {
        loadingScreen.classList.remove("active");
        submitBtn.disabled = false;
        showErrors(data.errors || ["Something went wrong submitting the files."]);
        return;
      }

      pollStatus(data.job_id);
    } catch (err) {
      loadingScreen.classList.remove("active");
      submitBtn.disabled = false;
      showErrors(["Could not reach the server. Is the GenoFold app running?"]);
    }
  });

  function pollStatus(jobId) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        const data = await res.json();

        if (!data.ok) {
          clearInterval(interval);
          return;
        }

        loadingStage.textContent = STAGE_LABELS[data.stage] || data.stage || "Working";
        loadingMessage.textContent = data.message || "";
        loadingBar.style.width = `${data.percent || 0}%`;
        loadingPercent.textContent = `${data.percent || 0}%`;
        growThread(data.percent || 0);

        if (data.status === "done") {
          clearInterval(interval);
          loadingMessage.textContent = "Structure ready — opening results";
          setTimeout(() => {
            window.location.href = `/results/${jobId}`;
          }, 700);
        } else if (data.status === "error") {
          clearInterval(interval);
          loadingError.style.display = "block";
          loadingError.textContent = data.message || "The pipeline failed.";
        }
      } catch (err) {
        clearInterval(interval);
      }
    }, 900);
  }

  const heroThread = document.getElementById("thread-path");
  const heroDots = document.getElementById("thread-dots");
  if (heroThread && heroThread.getTotalLength) {
    const len = heroThread.getTotalLength();
    for (let i = 0; i < 7; i++) {
      const pt = heroThread.getPointAtLength((len / 7) * i);
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", pt.x);
      c.setAttribute("cy", pt.y);
      c.setAttribute("r", 2.4);
      c.setAttribute("fill", i % 3 === 0 ? "#e8a33d" : "#4fb8a6");
      c.setAttribute("opacity", "0.85");
      heroDots.appendChild(c);
    }
  }

  const loadingSvg = document.getElementById("loading-svg");
  let threadNodes = [];
  const N_NODES = 40;

  function initThread() {
    loadingSvg.innerHTML = "";
    threadNodes = [];
    const w = 420, h = 220, cx = w / 2, cy = h / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("id", "live-thread-path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#357f73");
    path.setAttribute("stroke-width", "1.2");
    path.setAttribute("opacity", "0.55");
    loadingSvg.appendChild(path);

    const dotGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    dotGroup.setAttribute("id", "live-thread-dots");
    loadingSvg.appendChild(dotGroup);

    for (let i = 0; i < N_NODES; i++) {
      const angle = (i / N_NODES) * Math.PI * 5.2;
      const radius = 18 + (i / N_NODES) * 78;
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius * 0.7;
      threadNodes.push({ x, y, revealed: false });
    }
    renderThread(0);
  }

  function renderThread(percent) {
    const revealCount = Math.floor((percent / 100) * N_NODES);
    const path = document.getElementById("live-thread-path");
    const dotGroup = document.getElementById("live-thread-dots");
    if (!path || !dotGroup) return;

    dotGroup.innerHTML = "";
    let d = "";
    threadNodes.forEach((node, i) => {
      if (i > revealCount) return;
      d += (i === 0 ? "M" : "L") + node.x.toFixed(1) + "," + node.y.toFixed(1) + " ";

      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", node.x);
      dot.setAttribute("cy", node.y);
      const isRecent = i === revealCount;
      dot.setAttribute("r", isRecent ? 4 : 2.2);
      dot.setAttribute("fill", isRecent ? "#e8a33d" : "#4fb8a6");
      dot.setAttribute("opacity", isRecent ? "1" : "0.7");
      dotGroup.appendChild(dot);
    });
    path.setAttribute("d", d);
  }

  function growThread(percent) {
    renderThread(percent);
  }
})();
