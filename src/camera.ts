// Camera helpers — with special handling for virtual webcams (DroidCam, Iriun,
// EpocCam, OBS, Camo...). These drivers frequently hand the browser a *green*
// frame when:
//   • the requested resolution / facingMode isn't supported (over-constrained),
//   • the phone app isn't connected yet so the driver emits an empty YUV buffer,
//   • Chrome's hardware video decode mangles the NV12/I420 plane layout.
// We therefore (a) never use facingMode for a chosen device, (b) walk a ladder of
// progressively simpler constraints, and (c) actually *look* at the pixels and
// retry/warn when the frame is green, blank or frozen.

export interface CamDevice {
  deviceId: string;
  label: string;
  virtual: boolean;
}

/** Labels only appear after permission has been granted at least once. */
export async function listCameras(): Promise<CamDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((d) => d.kind === "videoinput")
    .map((d, i) => ({
      deviceId: d.deviceId,
      label: d.label || `Camera ${i + 1}`,
      virtual: /droidcam|iriun|epoccam|obs|virtual|ivcam|camo|ndi|manycam/i.test(d.label),
    }));
}

/**
 * Constraint ladder, easiest first. DroidCam (free) streams 640×480, so that is
 * attempt #1 — asking for 1280×720 is the single most common cause of a green
 * or black frame. `facingMode` is deliberately omitted: virtual cameras do not
 * implement it and an `exact` facingMode makes getUserMedia fail outright.
 */
export function constraintLadder(deviceId?: string): MediaStreamConstraints[] {
  const dev = deviceId ? { deviceId: { exact: deviceId } } : {};
  return [
    { video: { ...dev, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false },
    { video: { ...dev, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: { ...dev, width: { ideal: 320 }, height: { ideal: 240 } }, audio: false },
    { video: deviceId ? { ...dev } : true, audio: false },
  ];
}

/** Wait until the element has actually painted a few real frames. */
export async function waitForFrames(video: HTMLVideoElement, frames = 6, timeout = 4000) {
  const t0 = performance.now();
  while (video.readyState < 2 && performance.now() - t0 < timeout) {
    await new Promise((r) => setTimeout(r, 40));
  }
  const rvfc = (video as unknown as { requestVideoFrameCallback?: (cb: () => void) => number }).requestVideoFrameCallback;
  let seen = 0;
  await new Promise<void>((resolve) => {
    const done = () => resolve();
    const step = () => {
      if (++seen >= frames || performance.now() - t0 > timeout) return done();
      next();
    };
    const next = () => (rvfc ? rvfc.call(video, step) : requestAnimationFrame(step));
    next();
    setTimeout(done, timeout); // hard stop – some drivers never fire a frame
  });
  return video.videoWidth > 0 && video.videoHeight > 0;
}

export type FrameKind = "ok" | "green" | "blank" | "nosignal" | "frozen";
export interface FrameHealth {
  kind: FrameKind;
  signature: string;
  mean: [number, number, number];
}

const probe = document.createElement("canvas");
probe.width = 48;
probe.height = 36;

/** Downscale the current frame and classify it (green / blank / fine). */
export function inspectFrame(video: HTMLVideoElement): FrameHealth {
  const dead: FrameHealth = { kind: "nosignal", signature: "", mean: [0, 0, 0] };
  if (!video || !video.videoWidth || video.readyState < 2) return dead;

  const ctx = probe.getContext("2d", { willReadFrequently: true });
  if (!ctx) return dead;
  try {
    ctx.drawImage(video, 0, 0, probe.width, probe.height);
  } catch {
    return dead; // tainted / not ready
  }
  const { data } = ctx.getImageData(0, 0, probe.width, probe.height);
  const n = data.length / 4;
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
  r /= n; g /= n; b /= n;

  let varR = 0, varG = 0, varB = 0;
  for (let i = 0; i < data.length; i += 4) {
    varR += (data[i] - r) ** 2; varG += (data[i + 1] - g) ** 2; varB += (data[i + 2] - b) ** 2;
  }
  const sd = (Math.sqrt(varR / n) + Math.sqrt(varG / n) + Math.sqrt(varB / n)) / 3;
  const signature = `${r | 0}-${g | 0}-${b | 0}-${sd.toFixed(1)}`;
  const mean: [number, number, number] = [r, g, b];

  // Pure/near-pure green plane → the classic virtual-camera failure.
  const greenDominant = g - Math.max(r, b) > 22;
  if (greenDominant && sd < 42) return { kind: "green", signature, mean };
  // Flat colour (black / grey / white) with virtually no detail → no signal.
  if (sd < 2.5) return { kind: "blank", signature, mean };
  return { kind: "ok", signature, mean };
}

export function describeFrame(kind: FrameKind): string {
  switch (kind) {
    case "green":
      return "The camera is sending a green frame — this is a DroidCam / virtual-camera glitch, not your photo.";
    case "blank":
      return "The camera is sending an empty (solid-colour) frame. Is the phone app connected?";
    case "frozen":
      return "The camera feed is frozen — the virtual camera stopped sending new frames.";
    case "nosignal":
      return "No video signal from the selected camera.";
    default:
      return "";
  }
}

/** Turn a getUserMedia exception into something a human can act on. */
export function humanCamError(err: unknown): string {
  const e = err as { name?: string; message?: string } | null;
  switch (e?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera blocked. Allow camera access for this site (padlock icon → Camera → Allow) and retry.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera found. Connect your webcam (or start the DroidCam client) and press Refresh.";
    case "NotReadableError":
    case "TrackStartError":
      return "The camera is already in use by another app. Close Zoom / Teams / OBS — and make sure the DroidCam client window itself is connected but not previewing exclusively.";
    case "OverconstrainedError":
      return "This camera doesn't support the requested resolution. Press “Re-sync camera” to retry at 640×480.";
    default:
      return `Could not start the camera${e?.message ? `: ${e.message}` : "."}`;
  }
}

export const DROIDCAM_TIPS = [
  "Open DroidCam on the phone FIRST, then the PC/Mac client, and wait until it shows the live preview — the virtual camera outputs green until the phone is connected.",
  "Set the DroidCam client resolution to 640×480 (free version only supports that) and press “Re-sync camera” here.",
  "Pick the “DroidCam Source 3 / DroidCam Video” entry in the camera list — not “DroidCam Audio”.",
  "Only one app may read a virtual camera: quit Zoom, Teams, OBS, Discord, then retry.",
  "Chrome green-frame bug: open chrome://settings/system, turn OFF “Use graphics acceleration when available”, relaunch Chrome.",
  "Still green? Try Firefox, or switch the client to “DroidCam OBS” mode / reconnect over USB instead of Wi-Fi.",
];
