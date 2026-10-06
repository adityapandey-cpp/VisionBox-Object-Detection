// Detection engine.
// Primary path: in-browser inference with TensorFlow.js + COCO-SSD (works with no server).
// Fallback/optional path: the bundled Flask + TensorFlow + OpenCV backend (POST /api/detect)
// is used automatically when it is reachable and the user enables "server mode".
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

export type ModelKey = "fast" | "balanced" | "accurate";

export const MODELS: { key: ModelKey; name: string; base: cocoSsd.ObjectDetectionBaseModel; blurb: string }[] = [
  { key: "fast", name: "SSD MobileNet V2 (Lite)", base: "lite_mobilenet_v2", blurb: "Fastest · great for webcam" },
  { key: "balanced", name: "SSD MobileNet V1", base: "mobilenet_v1", blurb: "Balanced speed / accuracy" },
  { key: "accurate", name: "SSD MobileNet V2 (Full)", base: "mobilenet_v2", blurb: "Most accurate · slower" },
];

export interface Detection {
  label: string;
  class_id: number;
  score: number;
  /** normalized [ymin, xmin, ymax, xmax] */
  box: [number, number, number, number];
  id?: number;
}

const cache = new Map<ModelKey, Promise<cocoSsd.ObjectDetection>>();

/** Lazily load each model exactly once (promise cache == thread-safe-ish in JS). */
export function loadModel(key: ModelKey) {
  if (!cache.has(key)) {
    const base = MODELS.find((m) => m.key === key)!.base;
    cache.set(
      key,
      tf.ready().then(() => cocoSsd.load({ base }))
    );
  }
  return cache.get(key)!;
}

const CLASSES: string[] = [];
function classId(label: string) {
  let i = CLASSES.indexOf(label);
  if (i < 0) {
    CLASSES.push(label);
    i = CLASSES.length - 1;
  }
  return i;
}

export async function detect(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  opts: { model: ModelKey; threshold: number; maxDet: number }
): Promise<{ detections: Detection[]; inference_ms: number; width: number; height: number }> {
  const model = await loadModel(opts.model);
  const w = (input as HTMLVideoElement).videoWidth || (input as HTMLImageElement).naturalWidth || input.width;
  const h = (input as HTMLVideoElement).videoHeight || (input as HTMLImageElement).naturalHeight || input.height;
  const t0 = performance.now();
  const raw = await model.detect(input, opts.maxDet, opts.threshold);
  const inference_ms = performance.now() - t0;
  const detections: Detection[] = raw.map((r) => {
    const [x, y, bw, bh] = r.bbox;
    return {
      label: r.class,
      class_id: classId(r.class),
      score: r.score,
      box: [y / h, x / w, (y + bh) / h, (x + bw) / w] as [number, number, number, number],
    };
  });
  return { detections, inference_ms, width: w, height: h };
}

/* ---------------- centroid tracker (stable IDs across video frames) ---------------- */
export class CentroidTracker {
  private next = 1;
  private objects: { id: number; cx: number; cy: number; label: string; missed: number }[] = [];
  update(dets: Detection[]) {
    const used = new Set<number>();
    for (const d of dets) {
      const cx = (d.box[1] + d.box[3]) / 2;
      const cy = (d.box[0] + d.box[2]) / 2;
      let best: (typeof this.objects)[0] | null = null;
      let bestDist = 0.08; // normalized distance gate
      for (const o of this.objects) {
        if (used.has(o.id) || o.label !== d.label) continue;
        const dist = Math.hypot(o.cx - cx, o.cy - cy);
        if (dist < bestDist) { bestDist = dist; best = o; }
      }
      if (best) { best.cx = cx; best.cy = cy; best.missed = 0; used.add(best.id); d.id = best.id; }
      else { const o = { id: this.next++, cx, cy, label: d.label, missed: 0 }; this.objects.push(o); used.add(o.id); d.id = o.id; }
    }
    this.objects = this.objects.filter((o) => (used.has(o.id) ? true : ++o.missed < 10));
    return dets;
  }
  reset() { this.objects = []; this.next = 1; }
}

/* ---------------- drawing ---------------- */
const PALETTE = ["#38bdf8", "#f472b6", "#4ade80", "#fbbf24", "#a78bfa", "#fb7185", "#34d399", "#60a5fa", "#f97316", "#e879f9"];
export const colorFor = (label: string) => {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 9973;
  return PALETTE[h % PALETTE.length];
};

export function drawDetections(ctx: CanvasRenderingContext2D, dets: Detection[], w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(2, Math.round(Math.min(w, h) / 300));
  ctx.font = `${Math.max(12, Math.round(Math.min(w, h) / 36))}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const d of dets) {
    const x = d.box[1] * w, y = d.box[0] * h;
    const bw = (d.box[3] - d.box[1]) * w, bh = (d.box[2] - d.box[0]) * h;
    const c = colorFor(d.label);
    ctx.strokeStyle = c;
    ctx.strokeRect(x, y, bw, bh);
    const text = `${d.id ? `#${d.id} ` : ""}${d.label} ${(d.score * 100).toFixed(0)}%`;
    const m = ctx.measureText(text);
    const padY = parseInt(ctx.font, 10) + 6;
    const ty = y - padY < 0 ? y : y - padY;
    ctx.fillStyle = c;
    ctx.fillRect(x - ctx.lineWidth / 2, ty, m.width + 12, padY);
    ctx.fillStyle = "#0b1020";
    ctx.fillText(text, x + 5, ty + 3);
  }
}
