(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const nav = $("#navtop");
  window.addEventListener("scroll", () => nav?.classList.toggle("on", scrollY > 8), { passive: true });
  $("#burger")?.addEventListener("click", () => $("#navlinks")?.classList.toggle("open"));
  $$("#navlinks a").forEach((a) => a.addEventListener("click", () => $("#navlinks")?.classList.remove("open")));
  const io = new IntersectionObserver((ents) => {
    ents.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("show"); io.unobserve(e.target); } });
  }, { threshold: 0.1, rootMargin: "0px 0px -30px 0px" });
  $$(".fade").forEach((el) => io.observe(el));
  const modal = $("#modal"), mv = $("#modal-video"), mt = $("#modal-title");
  const open = (src, title) => {
    mt.textContent = title || "Preview"; mv.src = src; modal.classList.add("open");
    mv.play().catch(() => {}); document.body.style.overflow = "hidden";
  };
  const close = () => {
    modal.classList.remove("open"); mv.pause(); mv.removeAttribute("src"); document.body.style.overflow = "";
  };
  $$("[data-video]").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); open(el.dataset.video, el.dataset.title); }));
  $("#modal-close")?.addEventListener("click", close);
  modal?.addEventListener("click", (e) => e.target === modal && close());
  window.addEventListener("keydown", (e) => e.key === "Escape" && close());
  $("#contact-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.currentTarget, d = new FormData(f);
    const name = String(d.get("name") || "").trim();
    const email = String(d.get("email") || "").trim();
    const msg = String(d.get("message") || "").trim();
    if (!name || !email || !msg) return;
    const to = f.dataset.to || "dhirajch145@gmail.com";
    location.href = `mailto:${to}?subject=${encodeURIComponent(`VidMCP · ${name}`)}&body=${encodeURIComponent(`From: ${name} <${email}>\n\n${msg}`)}`;
    const st = $("#form-status"); if (st) st.textContent = "Opening your email app…";
  });
  $$("[data-copy]").forEach((btn) => btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(btn.dataset.copy || ""); const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => btn.textContent = o, 1200); } catch {}
  }));
})();
