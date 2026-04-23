document.addEventListener("DOMContentLoaded", () => {
    // ── THEME HANDLING ──
    const savedTheme = localStorage.getItem('oscc-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    if (savedTheme === 'light') {
        document.getElementById('svg-moon').style.display = 'none';
        document.getElementById('svg-sun').style.display = '';
    }

    document.getElementById('theme-btn').onclick = () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('oscc-theme', next);
        document.getElementById('svg-moon').style.display = next === 'dark' ? '' : 'none';
        document.getElementById('svg-sun').style.display = next === 'light' ? '' : 'none';
    };

    // ── AUTHENTICATION ──
    let _apiKey = '';

    document.getElementById('eye-btn').onclick = () => {
        const inp = document.getElementById('key-input');
        const open = document.getElementById('svg-eye-open');
        const closed = document.getElementById('svg-eye-closed');
        if (inp.type === 'password') {
            inp.type = 'text';
            open.style.display = 'none';
            closed.style.display = '';
        } else {
            inp.type = 'password';
            open.style.display = '';
            closed.style.display = 'none';
        }
    };

    function setAuthError(msg) {
        const el = document.getElementById('auth-error');
        el.textContent = msg;
        document.getElementById('key-input').classList.toggle('err', !!msg);
    }

    function unlockDashboard(key) {
        _apiKey = key;
        if (key) {
            const masked = key.slice(0, 4) + '••••' + key.slice(-4);
            document.getElementById('key-chip-text').textContent = masked;
            document.getElementById('key-chip').style.display = 'flex';
        }
        const gate = document.getElementById('auth-gate');
        gate.style.opacity = '0';
        setTimeout(() => { gate.style.display = 'none'; }, 400);
        document.getElementById('app').classList.add('on');
        checkHealth();
    }

    async function doAuth() {
        const key = document.getElementById('key-input').value.trim();
        const btn = document.getElementById('auth-btn');
        const lbl = document.getElementById('auth-btn-label');
        setAuthError('');
        btn.disabled = true;
        lbl.textContent = 'Verifying...';

        try {
            const headers = { 'Content-Type': 'application/json' };
            if (key) headers['X-API-Key'] = key;
            const r = await fetch('/api/auth/verify', { method: 'POST', headers });
            const d = await r.json();

            if (r.status === 403 || (r.ok && d.valid === false)) {
                setAuthError('Invalid API key — please try again.');
            } else if (r.status === 429) {
                setAuthError('Too many attempts. Please wait a moment.');
            } else if (!r.ok) {
                setAuthError('Server error. Is the app running?');
            } else {
                unlockDashboard(key);
                return;
            }
        } catch (e) {
            setAuthError('Cannot reach server. Is the app running?');
        }
        btn.disabled = false;
        lbl.textContent = 'Access Dashboard';
    }

    document.getElementById('auth-btn').onclick = doAuth;
    document.getElementById('key-input').onkeydown = e => { if (e.key === 'Enter') doAuth(); };

    // Auto-check auth
    (async function () {
        try {
            const r = await fetch('/api/auth/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
            const d = await r.json();
            if (!d.auth_required) unlockDashboard('');
        } catch (_) { }
    })();

    // ── UPLOAD HANDLING ──
    let _file = null;

    function showPreview(file) {
        _file = file;
        const reader = new FileReader();
        reader.onload = e => {
            const prev = document.getElementById('up-preview');
            prev.src = e.target.result;
            prev.style.display = 'block';
            document.getElementById('up-icon-wrap').style.display = 'none';
            document.getElementById('up-text').style.display = 'none';
            document.getElementById('up-formats').style.display = 'none';
        };
        reader.readAsDataURL(file);
        document.getElementById('classify-btn').disabled = false;
        document.getElementById('error-box').style.display = 'none';
        hideResults();
    }

    function clearAll() {
        _file = null;
        const prev = document.getElementById('up-preview');
        prev.src = ''; prev.style.display = 'none';
        document.getElementById('up-icon-wrap').style.display = 'flex';
        document.getElementById('up-text').style.display = 'block';
        document.getElementById('up-formats').style.display = 'block';
        document.getElementById('classify-btn').disabled = true;
        document.getElementById('error-box').style.display = 'none';
        document.getElementById('file-input').value = '';
        hideResults();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function hideResults() {
        ['result-panel', 'gradcam-panel', 'lime-panel'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
    }

    document.getElementById('file-input').onchange = e => { if (e.target.files[0]) showPreview(e.target.files[0]); };
    document.getElementById('clear-btn').onclick = clearAll;

    const dz = document.getElementById('drop-zone');
    dz.ondragover = e => { e.preventDefault(); };
    dz.ondrop = e => { e.preventDefault(); if (e.dataTransfer.files[0]) showPreview(e.dataTransfer.files[0]); };

    // ── CLASSIFICATION ──
    document.getElementById('classify-btn').onclick = async function () {
        if (!_file) return;
        const btn = this;
        btn.disabled = true;
        document.getElementById('spinner').style.display = 'block';
        document.getElementById('error-box').style.display = 'none';
        hideResults();

        try {
            const fd = new FormData();
            fd.append('file', _file);
            const headers = {};
            if (_apiKey) headers['X-API-Key'] = _apiKey;

            const r = await fetch('/api/predict', { method: 'POST', headers, body: fd });
            const data = await r.json();
            document.getElementById('spinner').style.display = 'none';
            btn.disabled = false;

            if (!r.ok || data.error || data.detail) {
                const msg = data.detail || data.error || r.statusText;
                const eb = document.getElementById('error-box');
                eb.textContent = 'Error: ' + msg;
                eb.style.display = 'block';
                return;
            }
            renderResults(data);
        } catch (e) {
            document.getElementById('spinner').style.display = 'none';
            btn.disabled = false;
            const eb = document.getElementById('error-box');
            eb.textContent = 'Network error — is the server running?';
            eb.style.display = 'block';
        }
    };

    function renderResults(data) {
        const { prediction, image, gradcam, lime } = data;
        const { class: cls, confidence, scores } = prediction;
        const isOSCC = cls === 'OSCC';
        const cc = isOSCC ? 'o' : 'n';

        document.getElementById('result-panel').style.display = 'block';
        const card = document.getElementById('result-card');
        card.className = 'card result-card ' + (isOSCC ? 'r-oscc' : 'r-normal');

        document.getElementById('r-dot').style.background = isOSCC ? 'var(--danger)' : 'var(--success)';
        document.getElementById('r-ts').textContent = new Date().toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
        document.getElementById('result-img').src = 'data:image/jpeg;base64,' + image;

        const vEl = document.getElementById('verdict');
        vEl.textContent = cls;
        vEl.className = 'verdict-title v-' + cc;

        document.getElementById('verdict-desc').textContent = isOSCC
            ? 'The model detected features consistent with Oral Squamous Cell Carcinoma.'
            : 'No OSCC features detected. Tissue appears histologically normal.';

        const fill = document.getElementById('conf-fill');
        fill.className = 'progress-fill f-' + cc;
        setTimeout(() => fill.style.width = confidence + '%', 60);

        const cv = document.getElementById('conf-val');
        cv.textContent = confidence.toFixed(1) + '%';
        cv.className = 'conf-value v-' + cc;

        const sb = document.getElementById('score-bars');
        sb.innerHTML = Object.entries(scores).map(([lbl, pct]) => {
            const sc = lbl === 'OSCC' ? 'o' : 'n';
            return `<div class="score-row">
                <span class="score-label">${lbl}</span>
                <div class="score-track"><div class="score-bar s-${sc}" style="width:0" data-pct="${pct}"></div></div>
                <span class="score-pct">${pct.toFixed(1)}%</span>
            </div>`;
        }).join('');
        setTimeout(() => sb.querySelectorAll('.score-bar').forEach(b => b.style.width = b.dataset.pct + '%'), 60);

        document.getElementById('gradcam-panel').style.display = 'block';
        document.getElementById('gradcam-img').src = 'data:image/png;base64,' + gradcam;
        document.getElementById('lime-panel').style.display = 'block';
        document.getElementById('lime-img').src = 'data:image/png;base64,' + lime;

        document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    async function checkHealth() {
        try {
            const r = await fetch('/api/health');
            const d = await r.json();
            const el = document.getElementById('model-status-txt');
            if (el) {
                el.textContent = d.model_loaded ? 'Model Ready' : 'Model Offline';
                el.style.color = d.model_loaded ? 'var(--success)' : 'var(--danger)';
            }
        } catch (_) { }
    }
});