const youtubeUrlInput = document.getElementById("youtubeUrl");
const tosConfirm = document.getElementById("tosConfirm");
const clipDurationSelect = document.getElementById("clipDuration");
const numClipsSelect = document.getElementById("numClips");
const subtitleEnabledSelect = document.getElementById("subtitleEnabled");
const subtitleLangSelect = document.getElementById("subtitleLang");
const smartCropSelect = document.getElementById("smartCrop");
const apiBaseUrlInput = document.getElementById("apiBaseUrl");

const generateBtn = document.getElementById("generateBtn");
const progressSection = document.getElementById("progressSection");
const statusText = document.getElementById("statusText");
const progressPercent = document.getElementById("progressPercent");
const progressBarFill = document.getElementById("progressBarFill");
const errorBox = document.getElementById("errorBox");
const resultsSection = document.getElementById("resultsSection");
const clipsGrid = document.getElementById("clipsGrid");
const downloadAllBtn = document.getElementById("downloadAllBtn");

let currentJobId = null;
let pollTimer = null;

// Subtitle OFF থাকলে Language dropdown নিষ্ক্রিয় করা
subtitleEnabledSelect.addEventListener("change", () => {
  subtitleLangSelect.disabled = subtitleEnabledSelect.value === "false";
});

function getApiBase() {
  const custom = apiBaseUrlInput.value.trim().replace(/\/+$/, "");
  if (custom) return custom;
  if (window.location.protocol.startsWith("http") && !window.location.hostname.includes("github.io")) {
    return window.location.origin;
  }
  return "http://localhost:8000";
}

function showError(msg) {
  errorBox.textContent = "❌ " + msg;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
}

function setLoading(isLoading) {
  generateBtn.disabled = isLoading;
  generateBtn.textContent = isLoading ? "⏳ প্রসেসিং চলছে..." : "✨ Generate Shorts";
}

generateBtn.addEventListener("click", async () => {
  hideError();
  resultsSection.classList.add("hidden");
  clipsGrid.innerHTML = "";

  const url = youtubeUrlInput.value.trim();
  if (!url || (!url.includes("youtube.com") && !url.includes("youtu.be"))) {
    showError("অনুগ্রহ করে একটি সঠিক YouTube ভিডিও লিংক দিন।");
    return;
  }

  if (!tosConfirm.checked) {
    showError("প্রসেসিং শুরু করার আগে কপিরাইট ও Terms of Service সম্মতি বক্সে টিক দিন।");
    return;
  }

  const payload = {
    youtube_url: url,
    clip_duration: parseInt(clipDurationSelect.value, 10),
    num_clips: parseInt(numClipsSelect.value, 10),
    subtitle_enabled: subtitleEnabledSelect.value === "true",
    subtitle_language: subtitleLangSelect.value,
    smart_crop: smartCropSelect.value === "true",
  };

  setLoading(true);
  progressSection.classList.remove("hidden");
  updateProgress(5, "সার্ভারে অনুরোধ পাঠানো হচ্ছে...");

  try {
    const apiBase = getApiBase();
    const res = await fetch(`${apiBase}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "সার্ভার থেকে ত্রুটি পাওয়া গেছে।");
    }

    currentJobId = data.job_id;
    startPolling(apiBase, currentJobId);
  } catch (err) {
    showError(err.message || "Backend সার্ভারের সাথে সংযোগ করা যাচ্ছে না।");
    setLoading(false);
    progressSection.classList.add("hidden");
  }
});

function updateProgress(percent, message) {
  progressBarFill.style.width = `${percent}%`;
  progressPercent.textContent = `${percent}%`;
  statusText.textContent = message;
}

function startPolling(apiBase, jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${apiBase}/api/status/${jobId}`);
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "স্ট্যাটাস পাওয়া যায়নি।");
      }

      updateProgress(data.progress || 0, data.message || "প্রসেস হচ্ছে...");

      if (data.status === "completed") {
        clearInterval(pollTimer);
        setLoading(false);
        renderClips(apiBase, jobId, data.clips || []);
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        setLoading(false);
        progressSection.classList.add("hidden");
        showError(data.message || "ভিডিও প্রসেস করতে সমস্যা হয়েছে।");
      }
    } catch (err) {
      clearInterval(pollTimer);
      setLoading(false);
      showError("সার্ভার সংযোগ বিচ্ছিন্ন হয়েছে: " + err.message);
    }
  }, 2500);
}

function renderClips(apiBase, jobId, clips) {
  if (!clips.length) {
    showError("কোনো উপযুক্ত ক্লিপ তৈরি করা যায়নি।");
    return;
  }

  clipsGrid.innerHTML = "";
  clips.forEach((clip, idx) => {
    const videoUrl = `${apiBase}${clip.video_url}`;
    const downloadUrl = `${apiBase}${clip.download_url}`;

    const card = document.createElement("div");
    card.className = "clip-card";
    card.innerHTML = `
      <video class="clip-video" controls playsinline preload="metadata" src="${videoUrl}"></video>
      <div class="clip-body">
        <div class="clip-meta">
          <span>Clip #${idx + 1} • ${Math.round(clip.duration)}s</span>
          <span>AI Score: ${clip.score}/100</span>
        </div>
        <div class="clip-title">${escapeHtml(clip.title)}</div>
        <p class="clip-reason">${escapeHtml(clip.reason)}</p>
        <a class="download-btn" href="${downloadUrl}" download>
          ⬇️ Download MP4
        </a>
      </div>
    `;
    clipsGrid.appendChild(card);
  });

  downloadAllBtn.onclick = () => {
    window.location.href = `${apiBase}/api/download-all/${jobId}`;
  };

  resultsSection.classList.remove("hidden");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML;
}
