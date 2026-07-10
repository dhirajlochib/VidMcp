(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const nav = $("#navtop");
  window.addEventListener("scroll", () => nav?.classList.toggle("on", scrollY > 8), { passive: true });
  $("#burger")?.addEventListener("click", () => $("#navlinks")?.classList.toggle("open"));
  $$("#navlinks a").forEach((a) => a.addEventListener("click", () => $("#navlinks")?.classList.remove("open")));

  const io = new IntersectionObserver(
    (ents) => {
      ents.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add("show");
        io.unobserve(e.target);
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
  );
  $$(".fade, .card-soft, .g-card, .flow-step, .cl-item").forEach((el) => {
    if (!el.classList.contains("fade")) el.classList.add("fade");
    io.observe(el);
  });

  const modal = $("#modal"),
    mv = $("#modal-video"),
    mt = $("#modal-title");
  const open = (src, title) => {
    mt.textContent = title || "Preview";
    mv.src = src;
    modal.classList.add("open");
    mv.play().catch(() => {});
    document.body.style.overflow = "hidden";
  };
  const close = () => {
    modal.classList.remove("open");
    mv.pause();
    mv.removeAttribute("src");
    document.body.style.overflow = "";
  };
  $$("[data-video]").forEach((el) =>
    el.addEventListener("click", (e) => {
      e.preventDefault();
      open(el.dataset.video, el.dataset.title);
    })
  );
  $("#modal-close")?.addEventListener("click", close);
  modal?.addEventListener("click", (e) => e.target === modal && close());
  window.addEventListener("keydown", (e) => e.key === "Escape" && close());

  $("#contact-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.currentTarget,
      d = new FormData(f);
    const name = String(d.get("name") || "").trim();
    const email = String(d.get("email") || "").trim();
    const msg = String(d.get("message") || "").trim();
    if (!name || !email || !msg) return;
    const to = f.dataset.to || "dhirajch145@gmail.com";
    location.href = `mailto:${to}?subject=${encodeURIComponent(`VidMCP · ${name}`)}&body=${encodeURIComponent(
      `From: ${name} <${email}>\n\n${msg}`
    )}`;
    const st = $("#form-status");
    if (st) st.textContent = "Opening your email app…";
  });

  async function copyText(btn, text) {
    try {
      await navigator.clipboard.writeText(text || "");
      const o = btn.textContent;
      btn.textContent = "Copied";
      btn.classList.add("ok");
      setTimeout(() => {
        btn.textContent = o;
        btn.classList.remove("ok");
      }, 1400);
    } catch {}
  }
  $$("[data-copy]").forEach((btn) => btn.addEventListener("click", () => copyText(btn, btn.dataset.copy || "")));

  const term = $("#install-term");
  if (term) {
    $$(".term-tab", term).forEach((tab) => {
      tab.addEventListener("click", () => {
        $$(".term-tab", term).forEach((t) => t.classList.remove("on"));
        $$(".term-panel", term).forEach((p) => p.classList.remove("on"));
        tab.classList.add("on");
        term.querySelector(`[data-panel="${tab.dataset.tab}"]`)?.classList.add("on");
      });
    });
  }

  // soft particle field behind hero
  const canvas = $("#hero-canvas");
  if (canvas && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const ctx = canvas.getContext("2d");
    let w, h, pts, raf;
    const resize = () => {
      w = canvas.width = innerWidth;
      h = canvas.height = Math.min(innerHeight * 0.95, 900);
      const n = Math.min(70, Math.floor((w * h) / 18000));
      pts = Array.from({ length: n }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.6 + 0.4,
      }));
    };
    const tick = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of pts) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        ctx.beginPath();
        ctx.fillStyle = "rgba(212,255,42,0.35)";
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i],
            b = pts[j];
          const dx = a.x - b.x,
            dy = a.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d < 120) {
            ctx.strokeStyle = `rgba(42,255,209,${0.08 * (1 - d / 120)})`;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    resize();
    tick();
    addEventListener("resize", resize, { passive: true });
  }
})();
