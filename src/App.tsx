import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MODELS, ModelKey, Detection, detect, loadModel, drawDetections, colorFor, CentroidTracker,
} from "./detector";
import {
  CamDevice, listCameras, constraintLadder, waitForFrames, inspectFrame,
  describeFrame, humanCamError, DROIDCAM_TIPS, FrameKind,
} from "./camera";

type Tab = "image" | "video" | "webcam";

export default function App() {
  const [tab, setTab] = useState<Tab>("image");
  const [model, setModel] = useState<ModelKey>("fast");
  const [threshold, setThreshold] = useState(0.5);
  const [maxDet, setMaxDet] = useState(20);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [dets, setDets] = useState<Detection[]>([]);
  const [stats, setStats] = useState({ ms: 0, fps: 0 });
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [imgUrl, setImgUrl] = useState<string>("");
  const [videoUrl, setVideoUrl] = useState<string>("");

  // webcam state
  const [camOn, setCamOn] = useState(false);
  const [camBusy, setCamBusy] = useState(false);
  const [cameras, setCameras] = useState<CamDevice[]>([]);
  const [camId, setCamId] = useState<string>("");
  const [camWarning, setCamWarning] = useState<string>("");
  const [showTips, setShowTips] = useState(false);
  const [camInfo, setCamInfo] = useState<string>("");

  const imgRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const camRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tracker = useRef(new CentroidTracker());
  const rafRef = useRef<number>(0);

  const visible = useMemo(() => dets.filter((d) => !hidden.has(d.label)), [dets, hidden]);
  const classes = useMemo(() => Array.from(new Set(dets.map((d) => d.label))).sort(), [dets]);
  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    visible.forEach((d) => (m[d.label] = (m[d.label] || 0) + 1));
    return Object.entries(m).sort((a, b) => b[1] - a[1]);
  }, [visible]);

  /* ---------------- model preload ---------------- */
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setStatus(`Loading ${MODELS.find((m) => m.key === model)!.name}…`);
    loadModel(model)
      .then(() => alive && (setStatus("Model ready"), setLoading(false)))
      .catch(() => alive && (setError("Could not load the model. Check your connection."), setLoading(false)));
    return () => { alive = false; };
  }, [model]);

  /* ---------------- device list ---------------- */
  const refreshCameras = useCallback(async () => {
    const list = await listCameras();
    setCameras(list);
    setCamId((cur) => {
      if (cur && list.some((d) => d.deviceId === cur)) return cur;
      // prefer a virtual cam (DroidCam etc.) when present – that's usually what the user wants
      return (list.find((d) => d.virtual) || list[0])?.deviceId || "";
    });
  }, []);

  useEffect(() => {
    refreshCameras();
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshCameras);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", refreshCameras);
  }, [refreshCameras]);

  /* ---------------- canvas painting ---------------- */
  const paint = useCallback((list: Detection[], w: number, h: number) => {
    const c = canvasRef.current;
    if (!c) return;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    drawDetections(c.getContext("2d")!, list, w, h);
  }, []);

  useEffect(() => {
    const src: HTMLImageElement | HTMLVideoElement | null =
      tab === "image" ? imgRef.current : tab === "video" ? videoRef.current : camRef.current;
    if (!src) return;
    const w = (src as HTMLVideoElement).videoWidth || (src as HTMLImageElement).naturalWidth;
    const h = (src as HTMLVideoElement).videoHeight || (src as HTMLImageElement).naturalHeight;
    if (w && h) paint(visible, w, h);
  }, [visible, tab, paint]);

  /* ---------------- image ---------------- */
  const runImage = useCallback(async () => {
    const el = imgRef.current;
    if (!el || !el.naturalWidth) return;
    setError(""); setLoading(true);
    try {
      const r = await detect(el, { model, threshold, maxDet });
      setDets(r.detections);
      setStats({ ms: r.inference_ms, fps: 1000 / r.inference_ms });
      paint(r.detections, r.width, r.height);
    } catch { setError("Detection failed on this image."); }
    finally { setLoading(false); }
  }, [model, threshold, maxDet, paint]);

  useEffect(() => { if (tab === "image" && imgUrl) runImage(); }, [imgUrl, runImage, tab]);

  const validate = (f: File, kinds: string[]) => {
    if (f.size > 16 * 1024 * 1024) return "File is larger than the 16 MB limit.";
    if (!kinds.some((k) => f.type.includes(k))) return `Unsupported file type: ${f.type || "unknown"}.`;
    return "";
  };
  const onImageFile = (f?: File) => {
    if (!f) return;
    const e = validate(f, ["jpeg", "jpg", "png", "webp"]);
    if (e) return setError(e);
    setError(""); setDets([]); setImgUrl(URL.createObjectURL(f)); setTab("image");
  };
  const onVideoFile = (f?: File) => {
    if (!f) return;
    const e = validate(f, ["mp4", "webm", "video"]);
    if (e) return setError(e);
    setError(""); setDets([]); tracker.current.reset(); setVideoUrl(URL.createObjectURL(f)); setTab("video");
  };

  /* ---------------- live loop ---------------- */
  const startLoop = useCallback((el: HTMLVideoElement, isCam = false) => {
    cancelAnimationFrame(rafRef.current);
    let busy = false, last = performance.now(), fps = 0;
    const tick = async () => {
      rafRef.current = requestAnimationFrame(tick);
      if (busy || el.readyState < 2 || el.paused || el.ended) return;
      // Don't waste inference on green / empty virtual-camera frames.
      if (isCam && inspectFrame(el).kind !== "ok") return;
      busy = true;
      try {
        const r = await detect(el, { model, threshold, maxDet });
        setDets(tracker.current.update(r.detections));
        const now = performance.now();
        fps = fps * 0.8 + (1000 / Math.max(1, now - last)) * 0.2;
        last = now;
        setStats({ ms: r.inference_ms, fps });
      } catch { /* skip frame */ }
      busy = false;
    };
    tick();
  }, [model, threshold, maxDet]);

  const stopLoop = () => cancelAnimationFrame(rafRef.current);
  useEffect(() => () => stopLoop(), []);

  /* ---------------- webcam open / close / re-sync ---------------- */
  const stopCam = useCallback(() => {
    stopLoop();
    const s = camRef.current?.srcObject as MediaStream | null;
    s?.getTracks().forEach((t) => t.stop());
    if (camRef.current) camRef.current.srcObject = null;
    setCamOn(false); setCamWarning(""); setCamInfo(""); setDets([]);
  }, []);

  const openCam = useCallback(async (deviceId?: string) => {
    setCamBusy(true); setError(""); setCamWarning("");
    stopLoop();
    const prev = camRef.current?.srcObject as MediaStream | null;
    prev?.getTracks().forEach((t) => t.stop());        // release before re-opening
    if (camRef.current) camRef.current.srcObject = null;

    const ladder = constraintLadder(deviceId);
    let lastErr: unknown = null;
    let lastKind: FrameKind = "nosignal";

    for (let i = 0; i < ladder.length; i++) {
      let stream: MediaStream | null = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia(ladder[i]);
      } catch (e) { lastErr = e; continue; }

      const el = camRef.current;
      if (!el) { stream.getTracks().forEach((t) => t.stop()); break; }
      el.srcObject = null;            // clean re-attach clears stale YUV buffers
      el.srcObject = stream;
      el.muted = true;
      try { await el.play(); } catch { /* autoplay guard */ }
      await waitForFrames(el, 8);

      const health = inspectFrame(el);
      lastKind = health.kind;
      const track = stream.getVideoTracks()[0];
      const s = track?.getSettings?.() || {};

      if (health.kind === "ok" || i === ladder.length - 1) {
        setCamInfo(`${track?.label || "camera"} · ${s.width || el.videoWidth}×${s.height || el.videoHeight}`);
        setCamOn(true);
        setCamBusy(false);
        tracker.current.reset();
        refreshCameras();                                  // labels are available now
        if (health.kind !== "ok") { setCamWarning(describeFrame(health.kind)); setShowTips(true); }
        startLoop(el, true);
        return;
      }
      // green / blank → drop this profile and try the next, simpler one
      stream.getTracks().forEach((t) => t.stop());
      el.srcObject = null;
      await new Promise((r) => setTimeout(r, 250));
    }

    setCamBusy(false);
    setCamOn(false);
    if (lastErr) setError(humanCamError(lastErr));
    else { setCamWarning(describeFrame(lastKind)); setShowTips(true); }
  }, [refreshCameras, startLoop]);

  const toggleCam = () => (camOn ? stopCam() : openCam(camId || undefined));

  /* Health monitor: detects green / frozen feeds while running. */
  useEffect(() => {
    if (!camOn) return;
    let same = 0, prevSig = "";
    const id = window.setInterval(() => {
      const el = camRef.current;
      if (!el) return;
      const h = inspectFrame(el);
      if (h.kind !== "ok") { setCamWarning(describeFrame(h.kind)); return; }
      if (h.signature === prevSig) {
        if (++same >= 4) { setCamWarning(describeFrame("frozen")); return; }
      } else { same = 0; prevSig = h.signature; }
      setCamWarning("");
    }, 1200);
    return () => clearInterval(id);
  }, [camOn]);

  /* restart the stream when another device is picked */
  const changeCamera = async (id: string) => {
    setCamId(id);
    if (camOn) await openCam(id);
  };

  /* ---------------- exports ---------------- */
  const exportPng = () => {
    const src: HTMLImageElement | HTMLVideoElement | null =
      tab === "image" ? imgRef.current : tab === "video" ? videoRef.current : camRef.current;
    const c = canvasRef.current;
    if (!src || !c) return;
    const out = document.createElement("canvas");
    out.width = c.width; out.height = c.height;
    const ctx = out.getContext("2d")!;
    ctx.drawImage(src as CanvasImageSource, 0, 0, out.width, out.height);
    ctx.drawImage(c, 0, 0);
    const a = document.createElement("a");
    a.href = out.toDataURL("image/png"); a.download = "detections.png"; a.click();
  };
  const download = (data: string, name: string, type: string) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([data], { type }));
    a.download = name; a.click();
  };
  const exportJson = () =>
    download(JSON.stringify({ model, threshold, detections: visible, inference_ms: stats.ms }, null, 2), "detections.json", "application/json");
  const exportCsv = () =>
    download(["label,class_id,score,ymin,xmin,ymax,xmax",
      ...visible.map((d) => [d.label, d.class_id, d.score.toFixed(4), ...d.box.map((v) => v.toFixed(4))].join(","))].join("\n"),
      "detections.csv", "text/csv");

  const Slider = (p: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string }) => (
    <label className="block">
      <span className="flex justify-between text-xs font-medium text-slate-400"><span>{p.label}</span><span className="tabular-nums text-sky-300">{p.fmt(p.value)}</span></span>
      <input type="range" min={p.min} max={p.max} step={p.step} value={p.value}
        onChange={(e) => p.onChange(parseFloat(e.target.value))}
        className="mt-2 w-full accent-sky-400 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400" />
    </label>
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-sky-500/30">
      <header className="border-b border-white/10 bg-slate-900/60 backdrop-blur sticky top-0 z-20">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-sky-400 to-fuchsia-500 grid place-items-center text-lg">◉</div>
          <div className="flex-1">
            <h1 className="font-semibold leading-tight">VisionBox · Object Detection</h1>
            <p className="text-xs text-slate-400">TensorFlow SSD · COCO 80 classes · Flask + OpenCV backend included</p>
          </div>
          <span className={`text-xs px-2 py-1 rounded-full border ${loading ? "border-amber-400/40 text-amber-300" : "border-emerald-400/40 text-emerald-300"}`}>
            {loading ? "Loading…" : status || "Ready"}
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 grid gap-6 lg:grid-cols-[1fr_340px]">
        <section className="space-y-4">
          <div role="tablist" aria-label="Input source" className="flex gap-2">
            {(["image", "video", "webcam"] as Tab[]).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                className={`px-4 py-2 rounded-lg text-sm capitalize border transition focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  tab === t ? "bg-sky-500 text-slate-950 border-sky-400 font-semibold" : "border-white/10 bg-white/5 hover:bg-white/10"}`}>
                {t}
              </button>
            ))}
          </div>

          {error && <div role="alert" className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>}

          {camWarning && (
            <div role="alert" className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
              <div className="flex flex-wrap items-center gap-2">
                <span className="flex-1 min-w-[12rem]">⚠️ {camWarning}</span>
                <button onClick={() => openCam(camId || undefined)} disabled={camBusy}
                  className="px-3 py-1.5 rounded-md bg-amber-400 text-slate-950 text-xs font-semibold hover:bg-amber-300 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-white">
                  {camBusy ? "Re-syncing…" : "Re-sync camera"}
                </button>
                <button onClick={() => setShowTips((v) => !v)} aria-expanded={showTips}
                  className="px-3 py-1.5 rounded-md border border-amber-400/50 text-xs hover:bg-amber-400/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-300">
                  {showTips ? "Hide fixes" : "How to fix"}
                </button>
              </div>
              {showTips && (
                <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-xs text-amber-100/90">
                  {DROIDCAM_TIPS.map((t) => <li key={t}>{t}</li>)}
                </ol>
              )}
            </div>
          )}

          <div className="relative rounded-2xl border border-white/10 bg-slate-900/60 overflow-hidden grid place-items-center min-h-[320px]"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (!f) return; f.type.startsWith("video") ? onVideoFile(f) : onImageFile(f); }}>
            {tab === "image" && (imgUrl
              ? <img ref={imgRef} src={imgUrl} alt="uploaded" onLoad={runImage} className="max-h-[70vh] w-auto" />
              : <Empty title="Drop an image here" sub="JPG, PNG or WebP · up to 16 MB" />)}

            {tab === "video" && (videoUrl
              ? <video ref={videoRef} src={videoUrl} controls playsInline className="max-h-[70vh] w-auto"
                  onPlay={(e) => startLoop(e.currentTarget)} onPause={stopLoop} onEnded={stopLoop} />
              : <Empty title="Drop an MP4 / WebM video" sub="Frames are sampled and boxes drawn live" />)}

            {tab === "webcam" && (
              <>
                <video ref={camRef} autoPlay playsInline muted className={`max-h-[70vh] w-auto ${camOn ? "" : "hidden"}`} />
                {!camOn && <Empty title={camBusy ? "Starting camera…" : "Webcam is off"} sub={camBusy ? "Negotiating a supported resolution" : "Start the camera for real-time detection"} />}
              </>
            )}

            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 m-auto h-full w-full max-h-[70vh] max-w-full object-contain" />
          </div>

          {tab === "webcam" && (
            <div className="flex flex-wrap items-center gap-2 rounded-xl border border-white/10 bg-slate-900/60 p-3">
              <label htmlFor="camsel" className="text-xs text-slate-400">Camera</label>
              <select id="camsel" value={camId} onChange={(e) => changeCamera(e.target.value)}
                className="flex-1 min-w-[12rem] rounded-lg border border-white/10 bg-slate-800 px-3 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400">
                {cameras.length === 0 && <option value="">Default camera (grant permission to list)</option>}
                {cameras.map((c) => (
                  <option key={c.deviceId} value={c.deviceId}>{c.virtual ? "📱 " : "🎥 "}{c.label}</option>
                ))}
              </select>
              <button onClick={refreshCameras} className="px-3 py-2 rounded-lg text-sm border border-white/10 bg-white/5 hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400">Refresh</button>
              <button onClick={() => openCam(camId || undefined)} disabled={camBusy}
                className="px-3 py-2 rounded-lg text-sm border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400">
                Re-sync camera
              </button>
              {camInfo && <span className="text-xs text-slate-500 w-full sm:w-auto">{camInfo}</span>}
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <label className="cursor-pointer px-4 py-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 text-sm focus-within:ring-2 focus-within:ring-sky-400">
              Upload image
              <input type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(e) => onImageFile(e.target.files?.[0])} />
            </label>
            <label className="cursor-pointer px-4 py-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 text-sm focus-within:ring-2 focus-within:ring-sky-400">
              Upload video
              <input type="file" accept="video/mp4,video/webm" className="sr-only" onChange={(e) => onVideoFile(e.target.files?.[0])} />
            </label>
            <button onClick={() => { setTab("webcam"); toggleCam(); }} disabled={camBusy}
              className="px-4 py-2 rounded-lg text-sm bg-sky-500 text-slate-950 font-semibold hover:bg-sky-400 disabled:opacity-60 focus:outline-none focus-visible:ring-2 focus-visible:ring-white">
              {camBusy ? "Starting…" : camOn ? "Stop webcam" : "Start webcam"}
            </button>
            <div className="flex-1" />
            <button onClick={exportPng} disabled={!dets.length} className="px-3 py-2 rounded-lg text-sm border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-40">PNG</button>
            <button onClick={exportJson} disabled={!dets.length} className="px-3 py-2 rounded-lg text-sm border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-40">JSON</button>
            <button onClick={exportCsv} disabled={!dets.length} className="px-3 py-2 rounded-lg text-sm border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-40">CSV</button>
          </div>
        </section>

        <aside className="space-y-4">
          <Card title="Model">
            <div className="space-y-2">
              {MODELS.map((m) => (
                <button key={m.key} onClick={() => setModel(m.key)}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    model === m.key ? "border-sky-400 bg-sky-500/15" : "border-white/10 bg-white/5 hover:bg-white/10"}`}>
                  <div className="font-medium">{m.name}</div>
                  <div className="text-xs text-slate-400">{m.blurb}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Parameters">
            <div className="space-y-4">
              <Slider label="Confidence threshold" value={threshold} min={0.1} max={0.95} step={0.05} onChange={setThreshold} fmt={(v) => `${Math.round(v * 100)}%`} />
              <Slider label="Max objects" value={maxDet} min={1} max={50} step={1} onChange={setMaxDet} fmt={(v) => `${v}`} />
            </div>
          </Card>

          <Card title="Results">
            <div className="grid grid-cols-3 gap-2 text-center">
              <Stat label="Objects" value={`${visible.length}`} />
              <Stat label="Inference" value={`${stats.ms.toFixed(0)} ms`} />
              <Stat label="FPS" value={stats.fps ? stats.fps.toFixed(1) : "—"} />
            </div>
            {classes.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {classes.map((c) => {
                  const off = hidden.has(c);
                  return (
                    <button key={c} aria-pressed={!off}
                      onClick={() => setHidden((s) => { const n = new Set(s); n.has(c) ? n.delete(c) : n.add(c); return n; })}
                      className={`text-xs px-2 py-1 rounded-full border transition focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${off ? "opacity-40 border-white/10" : "border-transparent"}`}
                      style={{ background: off ? "transparent" : colorFor(c) + "30", color: off ? "#94a3b8" : colorFor(c) }}>
                      {c}
                    </button>
                  );
                })}
              </div>
            )}
            <div className="mt-3 max-h-64 overflow-auto divide-y divide-white/5">
              {visible.length === 0 ? (
                <p className="py-6 text-center text-sm text-slate-400">Nothing detected yet.<br /><span className="text-xs">Try lowering the confidence threshold.</span></p>
              ) : visible.map((d, i) => (
                <div key={i} className="flex items-center gap-2 py-2 text-sm">
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: colorFor(d.label) }} />
                  <span className="flex-1 truncate">{d.id ? <span className="text-slate-500">#{d.id} </span> : null}{d.label}</span>
                  <span className="tabular-nums text-slate-400">{(d.score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </Card>

          {counts.length > 0 && (
            <Card title="Per-class counts">
              <div className="space-y-1.5">
                {counts.map(([c, n]) => (
                  <div key={c} className="flex items-center gap-2 text-xs">
                    <span className="w-20 truncate">{c}</span>
                    <div className="h-2 flex-1 rounded bg-white/5 overflow-hidden">
                      <div className="h-full rounded" style={{ width: `${(n / counts[0][1]) * 100}%`, background: colorFor(c) }} />
                    </div>
                    <span className="tabular-nums w-5 text-right">{n}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          <Card title="Phone as webcam (DroidCam)">
            <ol className="list-decimal space-y-1.5 pl-4 text-xs text-slate-400">
              {DROIDCAM_TIPS.slice(0, 4).map((t) => <li key={t}>{t}</li>)}
            </ol>
            <p className="mt-2 text-[11px] text-slate-500">
              A green picture always comes from the virtual-camera driver, never from the detector. This app now
              detects green/blank/frozen frames automatically and retries at a supported resolution.
            </p>
          </Card>
        </aside>
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-10 text-xs text-slate-500">
        Inference runs in your browser with TensorFlow.js (SSD/COCO). A production Flask + TensorFlow + OpenCV
        backend (<code>backend/app.py</code>, <code>GET /api/models</code>, <code>POST /api/detect</code>) ships with this project — see <code>README.md</code>.
      </footer>
    </div>
  );
}

const Card = (p: { title: string; children: React.ReactNode }) => (
  <section className="rounded-2xl border border-white/10 bg-slate-900/60 p-4">
    <h2 className="text-sm font-semibold mb-3 text-slate-200">{p.title}</h2>
    {p.children}
  </section>
);
const Stat = (p: { label: string; value: string }) => (
  <div className="rounded-lg bg-white/5 py-2">
    <div className="text-base font-semibold tabular-nums">{p.value}</div>
    <div className="text-[10px] uppercase tracking-wide text-slate-400">{p.label}</div>
  </div>
);
const Empty = (p: { title: string; sub: string }) => (
  <div className="text-center p-10">
    <div className="text-4xl mb-2">🖼️</div>
    <p className="font-medium">{p.title}</p>
    <p className="text-sm text-slate-400">{p.sub}</p>
  </div>
);
