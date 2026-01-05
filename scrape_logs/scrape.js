const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");

const OUT_DIR = path.resolve("downloads");
const LOG_FILE = path.join(OUT_DIR, "dcl_downloads.log");

// Base URL variants (in order of likelihood)
const BASES = [
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dcl_",
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dcl",
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dlc", // typo-ish fallback
  "https://police.ucmerced.edu/sites/g/files/ufvvjh1446/f/page/documents/dlc_", // typo-ish fallback
];

// Inclusive date range [2023-10-01 .. 2025-08-19]
const START = new Date(Date.UTC(2023, 7, 10)); // Oct (0-based month)
const END = new Date(Date.UTC(2025, 11, 18)); // Aug

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

        await fsp.writeFile(outfile, buf);
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
    `\n===== Run started ${new Date().toISOString()} =====\n`,
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
      await logLine(
        `✅ dcl_${canonical}.pdf saved (from ${res.url}, ${res.size} bytes). ` +
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
