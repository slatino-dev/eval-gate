import * as core from "@actions/core";
import * as github from "@actions/github";
import { execFileSync } from "child_process";

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
 */
async function run(): Promise<void> {
  try {
    const dataset = core.getInput("dataset", { required: true });
    const baseUrl = core.getInput("base_url") || "http://localhost:8765";
    const model = core.getInput("model") || "default";
    const baseline = core.getInput("baseline");
    const threshold = core.getInput("threshold") || "0.05";
    const out = core.getInput("out") || "evalgate-report.json";

    core.info(`Running evalgate on dataset: ${dataset}`);

    // Each input is a separate element — no shell interpolation occurs.
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

    try {
      // execFileSync does NOT spawn a shell — arguments are passed directly to
      // the OS, so shell metacharacters in any input cannot be interpreted.
      const output = execFileSync("evalgate", args, { encoding: "utf8" });
      core.info(output);
    } catch (err: unknown) {
      if (err && typeof err === "object" && "stdout" in err) {
        core.error(String((err as { stdout: unknown }).stdout));
      }
      throw new Error("evalgate run failed — regression detected or eval error");
    }

    core.setOutput("report", out);
    core.info("Eval gate passed.");
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    core.setFailed(message);
  }
}

run();
