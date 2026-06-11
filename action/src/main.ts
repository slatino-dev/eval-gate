/**
 * evalgate GitHub Action entry-point.
 *
 * Inputs (action.yml):
 *   dataset       — path to YAML eval dataset
 *   base_url      — OpenAI-compatible endpoint (default: mock server)
 *   model         — model name
 *   baseline      — path to JSON baseline (optional)
 *   threshold     — regression threshold (default: 0.05)
 *   out           — output JSON report path
 *
 * Security note: all user-supplied inputs are passed as discrete array elements
 * to execFileSync so no shell is involved and shell-metacharacter injection is
 * not possible.
 *
 * PR comment strategy: one STICKY comment per PR (identified by the
 * EVALGATE_MARKER sentinel). On subsequent runs the action finds and updates
 * the existing comment instead of creating a new one — keeps the PR clean.
 */

import * as core from "@actions/core";
import * as github from "@actions/github";
import { execFileSync } from "child_process";
import * as fs from "fs";

const COMMENT_MARKER = "<!-- evalgate-sticky-comment -->";

// ──────────────────────────────────────────────────────────────────────────────
// Report types (mirrors the JSON written by evalgate's write_json_report)
// ──────────────────────────────────────────────────────────────────────────────

interface RunSummary {
  run_id: string;
  dataset: string;
  model: string;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  scores?: Record<string, number>;
}

interface ScorerDrift {
  scorer: string;
  baseline_mean: number;
  candidate_mean: number;
  delta: number;
  ci_low: number;
  ci_high: number;
  regressed: boolean;
  improved: boolean;
  significant: boolean;
}

interface DriftResult {
  scorers: ScorerDrift[];
  regressed: boolean;
  missing_scorers: string[];
  new_scorers: string[];
}

interface EvalReport {
  summary: RunSummary;
  drift?: DriftResult;
}

// ──────────────────────────────────────────────────────────────────────────────
// Markdown report builder
// ──────────────────────────────────────────────────────────────────────────────

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function sign(value: number): string {
  return value >= 0 ? "+" : "";
}

function buildMarkdown(report: EvalReport, failed: boolean): string {
  const { summary, drift } = report;
  const status = failed ? "FAIL ❌" : "PASS ✅";
  const lines: string[] = [
    COMMENT_MARKER,
    `## Eval Gate — ${status}`,
    "",
    "| Field | Value |",
    "|-------|-------|",
    `| Run | \`${summary.run_id}\` |`,
    `| Dataset | ${summary.dataset} |`,
    `| Model | ${summary.model} |`,
    `| Pass rate | ${pct(summary.pass_rate)} (${summary.passed}/${summary.total}) |`,
  ];

  if (drift) {
    lines.push("");
    lines.push("### Per-scorer drift");
    lines.push("");
    lines.push("| Scorer | Score | Delta | CI | Verdict |");
    lines.push("|--------|-------|-------|----|---------|");
    for (const d of drift.scorers) {
      const deltaStr = `${sign(d.delta)}${d.delta.toFixed(3)}`;
      const ciStr = `[${d.ci_low.toFixed(3)}, ${d.ci_high.toFixed(3)}]`;
      let verdict: string;
      if (d.regressed) {
        verdict = "**REGRESSED** ❌";
      } else if (d.improved) {
        verdict = "improved ✅";
      } else if (d.significant) {
        verdict = "significant";
      } else {
        verdict = "ok (noise)";
      }
      lines.push(
        `| ${d.scorer} | ${d.candidate_mean.toFixed(3)} | ${deltaStr} | ${ciStr} | ${verdict} |`
      );
    }
    if (drift.missing_scorers.length > 0) {
      lines.push("");
      lines.push(
        `_Scorers in baseline but absent from candidate: ${drift.missing_scorers.join(", ")}_`
      );
    }
    if (drift.new_scorers.length > 0) {
      lines.push("");
      lines.push(`_New scorers not in baseline: ${drift.new_scorers.join(", ")}_`);
    }
  }

  lines.push("");
  lines.push(`_Updated by [evalgate](https://github.com/marketplace) at ${new Date().toUTCString()}_`);
  return lines.join("\n");
}

// ──────────────────────────────────────────────────────────────────────────────
// Sticky PR comment (create or update)
// ──────────────────────────────────────────────────────────────────────────────

async function upsertPrComment(
  token: string,
  body: string
): Promise<void> {
  const ctx = github.context;
  if (ctx.eventName !== "pull_request" && ctx.eventName !== "pull_request_target") {
    core.info("Not a pull_request event — skipping PR comment.");
    return;
  }

  const prNumber = ctx.payload.pull_request?.number;
  if (!prNumber) {
    core.warning("Could not determine PR number — skipping PR comment.");
    return;
  }

  const octokit = github.getOctokit(token);
  const { owner, repo } = ctx.repo;

  // List existing comments and look for our sticky marker.
  const { data: comments } = await octokit.rest.issues.listComments({
    owner,
    repo,
    issue_number: prNumber,
    per_page: 100,
  });

  const existing = comments.find((c) => c.body?.includes(COMMENT_MARKER));

  if (existing) {
    await octokit.rest.issues.updateComment({
      owner,
      repo,
      comment_id: existing.id,
      body,
    });
    core.info(`Updated existing evalgate PR comment (id=${existing.id}).`);
  } else {
    await octokit.rest.issues.createComment({
      owner,
      repo,
      issue_number: prNumber,
      body,
    });
    core.info("Created new evalgate PR comment.");
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

/**
 * The action entry-point.  Exported so tests can call it directly without
 * triggering it at module-scope (which would run on every import).
 */
export async function run(): Promise<void> {
  try {
    const dataset = core.getInput("dataset", { required: true });
    const baseUrl = core.getInput("base_url") || "http://localhost:8765";
    const model = core.getInput("model") || "default";
    const baseline = core.getInput("baseline");
    const threshold = core.getInput("threshold") || "0.05";
    const out = core.getInput("out") || "evalgate-report.json";
    const githubToken = core.getInput("github_token");

    core.info(`Running evalgate on dataset: ${dataset}`);

    // Build args as a discrete array — execFileSync does NOT spawn a shell,
    // so shell metacharacters in any value are never interpreted.
    const args: string[] = [
      "run",
      dataset,
      `--base-url=${baseUrl}`,
      `--model=${model}`,
      `--threshold=${threshold}`,
      `--out=${out}`,
    ];
    if (baseline) {
      args.push(`--baseline=${baseline}`);
    }

    core.info(`Executing: evalgate ${args.join(" ")}`);

    let evalFailed = false;
    try {
      const output = execFileSync("evalgate", args, { encoding: "utf8" });
      core.info(output);
    } catch (err: unknown) {
      evalFailed = true;
      if (err && typeof err === "object" && "stdout" in err) {
        const stdout = String((err as { stdout: unknown }).stdout);
        if (stdout) core.info(stdout);
      }
      if (err && typeof err === "object" && "stderr" in err) {
        const stderr = String((err as { stderr: unknown }).stderr);
        if (stderr) core.error(stderr);
      }
      // Don't throw yet — we still want to post the report and comment.
    }

    core.setOutput("report", out);

    // Parse the JSON report and emit a job summary.
    let report: EvalReport | null = null;
    if (fs.existsSync(out)) {
      try {
        report = JSON.parse(fs.readFileSync(out, "utf8")) as EvalReport;
      } catch {
        core.warning(`Could not parse report at ${out}`);
      }
    }

    if (report) {
      const md = buildMarkdown(report, evalFailed);

      // Write GitHub Step Summary.
      await core.summary.addRaw(md).write();

      // Post / update a sticky PR comment when a token is available.
      if (githubToken) {
        try {
          await upsertPrComment(githubToken, md);
        } catch (commentErr: unknown) {
          const msg = commentErr instanceof Error ? commentErr.message : String(commentErr);
          core.warning(`Could not post PR comment: ${msg}`);
        }
      } else {
        core.info("No github_token provided — skipping PR comment.");
      }
    }

    if (evalFailed) {
      throw new Error("evalgate detected a regression — gate FAILED.");
    }

    core.info("Eval gate passed.");
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    core.setFailed(message);
  }
}

// Only auto-run when this file is the process entry-point, not when imported
// by tests.  We detect this by checking whether the module was invoked
// directly (via the __filename heuristic used by Node's module system).
if (process.env["VITEST"] === undefined) {
  run();
}
