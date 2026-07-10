(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const nav = $("#navtop");
  window.addEventListener("scroll", () => nav?.classList.toggle("on", scrollY > 8), { passive: true });
  $("#burger")?.addEventListener("click", () => $("#navlinks")?.classList.toggle("open"));
  $$("#navlinks a").forEach((a) => a.addEventListener("click", () => $("#navlinks")?.classList.remove("open")));

  // staggered reveal
  const io = new IntersectionObserver(
    (ents) => {
      ents.forEach((e, i) => {
        if (!e.isIntersecting) return;
        const el = e.target;
        el.style.transitionDelay = `${Math.min(i * 40, 200)}ms`;
        el.classList.add("show");
        io.unobserve(el);
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
  );
  $$(".fade, .card-soft, .g-card, .flow-step").forEach((el) => {
    if (!el.classList.contains("fade")) el.classList.add("fade");
    io.observe(el);
  });

  // video modal
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

  // contact
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

  // copy buttons
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
  $$("[data-copy]").forEach((btn) =>
    btn.addEventListener("click", () => copyText(btn, btn.dataset.copy || ""))
  );

  // install terminal tabs
  const term = $("#install-term");
  if (term) {
    $$(".term-tab", term).forEach((tab) => {
      tab.addEventListener("click", () => {
        $$(".term-tab", term).forEach((t) => t.classList.remove("on"));
        $$(".term-panel", term).forEach((p) => p.classList.remove("on"));
        tab.classList.add("on");
        const panel = term.querySelector(`[data-panel="${tab.dataset.tab}"]`);
        panel?.classList.add("on");
      });
    });
  }
})();
