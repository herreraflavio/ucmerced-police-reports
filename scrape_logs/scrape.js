const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");

const OUT_DIR = path.resolve("downloads");
const LOG_FILE = path.join(OUT_DIR, "dcl_downloads.log");

// Keep a stable "latest" copy for OCR
const LATEST_DIR = path.join(OUT_DIR, "latest");
const LATEST_PDF = path.join(LATEST_DIR, "latest.pdf");
const LATEST_META = path.join(LATEST_DIR, "latest.json");

// Base URL variants (in order of likelihood)
const BASES = [
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dcl_",
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dcl",
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dlc", // typo-ish fallback
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dlc_", // typo-ish fallback
];

// 1-day inclusive range: previous day from current date (UTC calendar day)
// const now = new Date();
// const YESTERDAY_UTC = new Date(
//   Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
// );
// YESTERDAY_UTC.setUTCDate(YESTERDAY_UTC.getUTCDate() - 1);

// const START = new Date(YESTERDAY_UTC);
// const END = new Date(YESTERDAY_UTC);
const now = new Date();

// choose how many days back you want (1 = yesterday, 2 = two days ago, etc.)
const daysBack = 2;

// Start of *today* in UTC (00:00:00.000)
const todayUTC = new Date(Date.UTC(
  now.getUTCFullYear(),
  now.getUTCMonth(),
  now.getUTCDate()
));

// Start of the target day in UTC
const START = new Date(todayUTC);
START.setUTCDate(START.getUTCDate() - daysBack);

// End is exactly 1 day later (exclusive)
const END = new Date(START);
END.setUTCDate(END.getUTCDate() + 1);

console.log({ START: START.toISOString(), END: END.toISOString() });


function fmt2(n) {
  return String(n).padStart(2, "0");
}
function stampCanonical(dUTC) {
  // Always for output file naming: MMDDYYYY
  return `${fmt2(dUTC.getUTCMonth() + 1)}${fmt2(
    dUTC.getUTCDate()
  )}${dUTC.getUTCFullYear()}`;
}

// Generate stamp candidates for a given date, deduped & prioritized.
// Examples for 12/04/2024 -> 12042024, 120424, 1242024, 12424, etc.
function stampCandidates(dUTC) {
  const m = dUTC.getUTCMonth() + 1;
  const day = dUTC.getUTCDate();
  const yyyy = dUTC.getUTCFullYear();
  const yy = String(yyyy).slice(-2);

  const MM = fmt2(m);
  const M = String(m);
  const DD = fmt2(day);
  const D = String(day);

  const out = [];
  const push = (s) => {
    if (!out.includes(s)) out.push(s);
  };

  // Most likely first:
  push(`${MM}${DD}${yyyy}`); // 08192025
  push(`${MM}${DD}${yy}`); // 081925

  // Remove leading zero from day
  push(`${MM}${D}${yyyy}`); // 1242024
  push(`${MM}${D}${yy}`); // 12424

  // Remove leading zero from month
  push(`${M}${DD}${yyyy}`); // 9122024
  push(`${M}${DD}${yy}`); // 91224

  // Remove both leading zeros
  push(`${M}${D}${yyyy}`); // 142024
  push(`${M}${D}${yy}`); // 1424

  return out;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function logLine(line) {
  const ts = new Date().toISOString();
  const full = `[${ts}] ${line}\n`;
  process.stdout.write(full);
  await fsp.appendFile(LOG_FILE, full, "utf8");
}

function daysInclusiveUTC(a, b) {
  const dayMs = 24 * 60 * 60 * 1000;
  const aMs = Date.UTC(a.getUTCFullYear(), a.getUTCMonth(), a.getUTCDate());
  const bMs = Date.UTC(b.getUTCFullYear(), b.getUTCMonth(), b.getUTCDate());
  return Math.floor((bMs - aMs) / dayMs) + 1;
}

function pct(n, d) {
  if (!d) return "0.00%";
  return ((n / d) * 100).toFixed(2) + "%";
}

function looksLikePdf(buf) {
  // Quick signature check: "%PDF-"
  if (!buf || buf.length < 5) return false;
  return buf.subarray(0, 5).toString("ascii") === "%PDF-";
}

async function atomicWriteFile(destPath, buf) {
  const tmpPath = `${destPath}.tmp.${process.pid}.${Date.now()}`;
  await fsp.writeFile(tmpPath, buf);
  await fsp.rename(tmpPath, destPath);
}

async function updateLatestFiles({ outfile, canonical, url, size }) {
  await fsp.mkdir(LATEST_DIR, { recursive: true });

  const meta = {
    canonical, // MMDDYYYY
    source_file: path.resolve(outfile),
    latest_pdf: path.resolve(LATEST_PDF),
    url,
    size,
    downloaded_at: new Date().toISOString(),
  };
  await fsp.writeFile(LATEST_META, JSON.stringify(meta, null, 2), "utf8");

  // Atomic replace of latest.pdf (copy to temp in same dir, then rename)
  const tmp = path.join(
    LATEST_DIR,
    `.latest.${process.pid}.${Date.now()}.tmp.pdf`
  );
  await fsp.copyFile(outfile, tmp);
  await fsp.rename(tmp, LATEST_PDF);
}

// Try all stamp variants × base URLs for a given date.
// Returns { ok, url?, size?, stampsTried, combosTried }
async function tryDownload(dUTC, outfile) {
  const headers = {
    "user-agent":
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    accept: "application/pdf,*/*;q=0.8",
  };

  const stamps = stampCandidates(dUTC);
  let combosTried = 0;

  for (const stamp of stamps) {
    for (const base of BASES) {
      const url = `${base}${stamp}.pdf`;
      combosTried++;
      try {
        const res = await fetch(url, { redirect: "follow", headers });
        if (!res.ok) continue;

        const buf = Buffer.from(await res.arrayBuffer());
        const ct = (res.headers.get("content-type") || "").toLowerCase();

        // Guard against HTML/tiny responses
        if (!ct.includes("pdf") || buf.length < 2048) continue;

        // Extra guard: PDF header signature
        if (!looksLikePdf(buf)) continue;

        // Atomic write for the archive file
        await atomicWriteFile(outfile, buf);

        return {
          ok: true,
          url,
          size: buf.length,
          stampsTried: stamps.length,
          combosTried,
        };
      } catch {
        // network error -> try next combo
      }
    }
  }
  return { ok: false, stampsTried: stamps.length, combosTried };
}

(async () => {
  await fsp.mkdir(OUT_DIR, { recursive: true });
  await fsp.appendFile(
    LOG_FILE,
    `\n===== Run started ${new Date().toISOString()} =====\n` +
      `Range: ${START.toISOString().slice(0, 10)} to ${END
        .toISOString()
        .slice(0, 10)} (UTC)\n`,
    "utf8"
  );

  const totalDays = daysInclusiveUTC(START, END);
  let dayIndex = 0;

  // Fail-rate counters (skip does not count as an attempt)
  let attempts = 0;
  let successes = 0;
  let failures = 0;

  for (let d = new Date(START); d <= END; d.setUTCDate(d.getUTCDate() + 1)) {
    dayIndex++;
    const canonical = stampCanonical(d);
    const outfile = path.join(OUT_DIR, `dcl_${canonical}.pdf`);

    if (fs.existsSync(outfile)) {
      await logLine(
        `↩️ dcl_${canonical}.pdf already exists (day ${dayIndex}/${totalDays}). ` +
          `Progress: attempts ${attempts}, success ${successes}, fail ${failures}, fail rate ${pct(
            failures,
            attempts
          )}`
      );
      continue;
    }

    const res = await tryDownload(d, outfile);
    attempts++;

    if (res.ok) {
      successes++;

      // Update downloads/latest/latest.pdf + latest.json
      await updateLatestFiles({
        outfile,
        canonical,
        url: res.url,
        size: res.size,
      });

      await logLine(
        `✅ dcl_${canonical}.pdf saved (from ${res.url}, ${res.size} bytes). ` +
          `Updated latest -> ${LATEST_PDF}. ` +
          `Day ${dayIndex}/${totalDays}. Attempts ${attempts}, success ${successes}, fail ${failures}, ` +
          `fail rate ${pct(failures, attempts)}`
      );
    } else {
      failures++;
      await logLine(
        `❌ ${canonical} not found (tried ${res.stampsTried} stamp variants × ${BASES.length} bases = ${res.combosTried} combos). ` +
          `Day ${dayIndex}/${totalDays}. Attempts ${attempts}, success ${successes}, fail ${failures}, ` +
          `fail rate ${pct(failures, attempts)}`
      );
    }

    await sleep(200); // be polite
  }

  await logLine("Done.");
})();
