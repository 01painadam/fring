// Headless Chrome check of the web app (mistakes.md M28).
//
// naga (native wgpu) accepts some WGSL that Tint (Chrome) rejects, so the
// native tests cannot show that the page works. This script opens the page in
// headless Chrome with WebGPU, reads the status line every 5 s, collects the
// console, and saves a screenshot. It fails (exit 1) when the simulated time
// does not advance or the console has a WGSL or GPU validation error.
//
// Usage (Node 22 or later, Google Chrome installed):
//   python3 -m http.server 8080 -d apps/fring-web/dist &
//   node apps/fring-web/scripts/cdp-check.mjs [url] [seconds] [screenshot.png]
// Defaults: http://localhost:8080/, 20 s, cdp-check.png in the current folder.
// Use "?bench" in the URL to run the in-page benchmark (about 80 s).

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const url = process.argv[2] ?? "http://localhost:8080/";
const secs = Number(process.argv[3] ?? 20);
const shotPath = process.argv[4] ?? "cdp-check.png";
const chromePath =
  process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9300 + Math.floor(Math.random() * 600);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "fring-cdp-"));

const chrome = spawn(
  chromePath,
  [
    "--headless=new",
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    "--window-size=1200,800",
    "about:blank",
  ],
  { stdio: "ignore" },
);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets = null;
for (let k = 0; k < 30 && !targets; k++) {
  await sleep(1000);
  try {
    targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  } catch {
    // Chrome is not ready yet.
  }
}
if (!targets) {
  console.error("Chrome did not open the debugging port in 30 s");
  chrome.kill();
  process.exit(2);
}

const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = {};
const lines = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending[m.id]) {
    pending[m.id](m.result);
    delete pending[m.id];
    return;
  }
  if (m.method === "Runtime.consoleAPICalled")
    lines.push(`console.${m.params.type}: ` + m.params.args.map((a) => a.value ?? a.description).join(" "));
  else if (m.method === "Runtime.exceptionThrown")
    lines.push("exception: " + (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
  else if (m.method === "Log.entryAdded") lines.push(`log.${m.params.entry.level}: ${m.params.entry.text}`);
};
const send = (method, params = {}) =>
  new Promise((r) => {
    const i = ++id;
    pending[i] = r;
    ws.send(JSON.stringify({ id: i, method, params }));
  });

await send("Runtime.enable");
await send("Log.enable");
await send("Page.enable");
await send("Page.navigate", { url });

const times = [];
for (let s = 0; s < secs; s += 5) {
  await sleep(5000);
  const r = await send("Runtime.evaluate", {
    expression: "document.getElementById('readout')?.textContent ?? ''",
  });
  const text = (r.result.value ?? "").replace(/\s+/g, " ").trim();
  lines.push(`t=${s + 5}s readout: ${text}`);
  // "t = 4.8 d of 9.99 y" or "t = 2.04 y of 9.99 y"
  const m = text.match(/t = ([\d.]+) (d|y) of/);
  if (m) times.push(Number(m[1]) * (m[2] === "y" ? 365.25 : 1));
}

const shot = await send("Page.captureScreenshot", { format: "png" });
fs.writeFileSync(shotPath, Buffer.from(shot.data, "base64"));
ws.close();
const exited = new Promise((r) => chrome.once("exit", r));
chrome.kill();
await Promise.race([exited, sleep(5000)]);
try {
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 3 });
} catch {
  console.log(`(could not remove the Chrome profile ${profile})`);
}

console.log(lines.join("\n"));
const gpuErrors = lines.filter((l) => /WGSL|ShaderModule|ComputePipeline|RenderPipeline|GPUValidationError|GPU rejected/i.test(l));
const advanced = times.length >= 2 && times.some((t, i) => i > 0 && t !== times[i - 1]);
console.log(`\nscreenshot: ${shotPath}`);
console.log(`simulated time advances: ${advanced ? "yes" : "NO"}`);
console.log(`WGSL or GPU errors: ${gpuErrors.length}`);
process.exit(advanced && gpuErrors.length === 0 ? 0 : 1);
