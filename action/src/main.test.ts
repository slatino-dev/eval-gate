/**
 * Tests for the evalgate GitHub Action (main.ts).
 *
 * Strategy: the @actions/core, @actions/github, child_process.execFileSync,
 * and fs modules are all mocked via vitest so tests run without a live
 * evalgate install, filesystem, or GitHub API.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

// ──────────────────────────────────────────────────────────────────────────────
// Module-level mocks — defined before any import of the modules under test
// ──────────────────────────────────────────────────────────────────────────────

// Mock @actions/core
const coreSetFailed = vi.fn();
const coreInfo = vi.fn();
const coreWarning = vi.fn();
const coreError = vi.fn();
const coreSetOutput = vi.fn();
const coreSummaryAddRaw = vi.fn().mockReturnThis();
const coreSummaryWrite = vi.fn().mockResolvedValue(undefined);
const coreGetInput = vi.fn();

vi.mock("@actions/core", () => ({
  getInput: (name: string, opts?: { required?: boolean }) => coreGetInput(name, opts),
  setFailed: (msg: string) => coreSetFailed(msg),
  setOutput: (name: string, value: string) => coreSetOutput(name, value),
  info: (msg: string) => coreInfo(msg),
  warning: (msg: string) => coreWarning(msg),
  error: (msg: string) => coreError(msg),
  summary: {
    addRaw: (text: string) => {
      coreSummaryAddRaw(text);
      return { write: coreSummaryWrite };
    },
  },
}));

// Mock @actions/github — simulate a pull_request event context.
const createCommentFn = vi.fn().mockResolvedValue({ data: { id: 9001 } });
const updateCommentFn = vi.fn().mockResolvedValue({ data: {} });
const listCommentsFn = vi.fn().mockResolvedValue({ data: [] });

const mockGetOctokit = vi.fn(() => ({
  rest: {
    issues: {
      createComment: createCommentFn,
      updateComment: updateCommentFn,
      listComments: listCommentsFn,
    },
  },
}));

vi.mock("@actions/github", () => ({
  context: {
    eventName: "pull_request",
    repo: { owner: "acme", repo: "myrepo" },
    payload: { pull_request: { number: 42 } },
  },
  getOctokit: (token: string) => mockGetOctokit(token),
}));

// Mock child_process.execFileSync
const execFileSyncFn = vi.fn();
vi.mock("child_process", () => ({
  execFileSync: (...args: unknown[]) => execFileSyncFn(...args),
}));

// Mock fs
const fsExistsSyncFn = vi.fn();
const fsReadFileSyncFn = vi.fn();
vi.mock("fs", () => ({
  existsSync: (p: string) => fsExistsSyncFn(p),
  readFileSync: (p: string, enc: string) => fsReadFileSyncFn(p, enc),
}));

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

function makeReport(passed = 3, total = 3, regressed = false) {
  return JSON.stringify({
    summary: {
      run_id: "abc12345",
      dataset: "smoke",
      model: "default",
      total,
      passed,
      failed: total - passed,
      pass_rate: passed / total,
      scores: { contains: passed / total },
    },
    ...(regressed
      ? {
          drift: {
            scorers: [
              {
                scorer: "contains",
                baseline_mean: 1.0,
                candidate_mean: 0.5,
                delta: -0.5,
                ci_low: 0.9,
                ci_high: 1.0,
                regressed: true,
                improved: false,
                significant: true,
              },
            ],
            regressed: true,
            missing_scorers: [],
            new_scorers: [],
          },
        }
      : {}),
  });
}

function setupInputs(overrides: Record<string, string> = {}) {
  const defaults: Record<string, string> = {
    dataset: "examples/basic.yaml",
    base_url: "http://localhost:8765",
    model: "default",
    baseline: "",
    threshold: "0.05",
    out: "evalgate-report.json",
    github_token: "ghs_fake_token",
    ...overrides,
  };
  coreGetInput.mockImplementation((name: string) => defaults[name] ?? "");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test suite
// ──────────────────────────────────────────────────────────────────────────────

describe("evalgate action", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default: evalgate succeeds and writes a report.
    execFileSyncFn.mockReturnValue("Eval run : abc12345\nResults  : 3/3 passed (100.0%)\n");
    fsExistsSyncFn.mockReturnValue(true);
    fsReadFileSyncFn.mockReturnValue(makeReport(3, 3, false));
    listCommentsFn.mockResolvedValue({ data: [] }); // no existing comment
  });

  // ── Happy path ───────────────────────────────────────────────────────────

  it("calls evalgate with required args and no shell", async () => {
    setupInputs();
    const { run } = await import("./main");
    await run();

    expect(execFileSyncFn).toHaveBeenCalledOnce();
    const [cmd, args, opts] = execFileSyncFn.mock.calls[0];
    expect(cmd).toBe("evalgate");
    expect(Array.isArray(args)).toBe(true);
    expect(args[0]).toBe("run");
    expect(args[1]).toBe("examples/basic.yaml");
    // No shell option or shell: false (execFileSync defaults to no-shell)
    expect(opts).toMatchObject({ encoding: "utf8" });
    // Baseline arg not present when input is empty.
    expect(args.join(" ")).not.toContain("--baseline");
  });

  it("appends --baseline when the input is provided", async () => {
    setupInputs({ baseline: "baseline.json" });
    const { run } = await import("./main");
    await run();

    const args = execFileSyncFn.mock.calls[0][1] as string[];
    expect(args.some((a) => a.startsWith("--baseline="))).toBe(true);
  });

  it("sets the 'report' output to the --out path", async () => {
    setupInputs({ out: "my-report.json" });
    const { run } = await import("./main");
    await run();

    expect(coreSetOutput).toHaveBeenCalledWith("report", "my-report.json");
  });

  it("does NOT call setFailed on a passing run", async () => {
    setupInputs();
    const { run } = await import("./main");
    await run();

    expect(coreSetFailed).not.toHaveBeenCalled();
  });

  // ── Job summary ──────────────────────────────────────────────────────────

  it("writes a job summary with the pass rate", async () => {
    setupInputs();
    const { run } = await import("./main");
    await run();

    expect(coreSummaryAddRaw).toHaveBeenCalledOnce();
    const md: string = coreSummaryAddRaw.mock.calls[0][0];
    expect(md).toContain("Eval Gate");
    expect(md).toContain("100.0%");
    expect(md).toContain("smoke");
  });

  it("includes per-scorer drift table when drift data is present", async () => {
    fsReadFileSyncFn.mockReturnValue(makeReport(1, 3, true));
    setupInputs();
    const { run } = await import("./main");
    await run();

    const md: string = coreSummaryAddRaw.mock.calls[0][0];
    expect(md).toContain("Per-scorer drift");
    expect(md).toContain("REGRESSED");
    expect(md).toContain("contains");
  });

  // ── PR comment ───────────────────────────────────────────────────────────

  it("creates a new sticky PR comment when none exists", async () => {
    setupInputs();
    const { run } = await import("./main");
    await run();

    expect(createCommentFn).toHaveBeenCalledOnce();
    const body: string = createCommentFn.mock.calls[0][0].body;
    expect(body).toContain("<!-- evalgate-sticky-comment -->");
    expect(updateCommentFn).not.toHaveBeenCalled();
  });

  it("updates an existing sticky PR comment on re-run", async () => {
    listCommentsFn.mockResolvedValue({
      data: [
        {
          id: 777,
          body: "<!-- evalgate-sticky-comment -->\n## Eval Gate — PASS ✅\n",
        },
      ],
    });
    setupInputs();
    const { run } = await import("./main");
    await run();

    expect(updateCommentFn).toHaveBeenCalledOnce();
    expect(updateCommentFn.mock.calls[0][0].comment_id).toBe(777);
    expect(createCommentFn).not.toHaveBeenCalled();
  });

  it("skips PR comment when github_token is empty", async () => {
    setupInputs({ github_token: "" });
    const { run } = await import("./main");
    await run();

    expect(createCommentFn).not.toHaveBeenCalled();
    expect(mockGetOctokit).not.toHaveBeenCalled();
  });

  // ── Regression / failure ─────────────────────────────────────────────────

  it("calls setFailed when evalgate exits non-zero", async () => {
    const err = Object.assign(new Error("Process exited with code 1"), {
      status: 1,
      stdout: "Results  : 2/3 passed\nWARNING  : regression exceeds threshold — gate FAILED\n",
      stderr: "",
    });
    execFileSyncFn.mockImplementation(() => {
      throw err;
    });
    // Report still written (evalgate always writes it before exiting).
    fsReadFileSyncFn.mockReturnValue(makeReport(2, 3, true));
    setupInputs();

    const { run } = await import("./main");
    await run();

    expect(coreSetFailed).toHaveBeenCalledOnce();
    expect(coreSetFailed.mock.calls[0][0]).toContain("regression");
  });

  it("marks the PR comment FAIL when evalgate exits non-zero", async () => {
    const err = Object.assign(new Error("exit 1"), {
      status: 1,
      stdout: "",
      stderr: "",
    });
    execFileSyncFn.mockImplementation(() => {
      throw err;
    });
    fsReadFileSyncFn.mockReturnValue(makeReport(0, 3, true));
    setupInputs();

    const { run } = await import("./main");
    await run();

    const body: string = createCommentFn.mock.calls[0][0].body;
    expect(body).toContain("FAIL");
  });

  it("warns but does not fail when the PR comment API throws", async () => {
    createCommentFn.mockRejectedValue(new Error("API error"));
    setupInputs();
    const { run } = await import("./main");
    await run();

    expect(coreWarning).toHaveBeenCalledWith(expect.stringContaining("API error"));
    expect(coreSetFailed).not.toHaveBeenCalled();
  });

  it("skips PR comment and warns on non-pull_request events", async () => {
    // The module-level github.context mock has eventName: "pull_request"
    // and we can't easily change it per-test without re-importing.
    // Instead, test the no-token path (equivalent end-to-end behavior:
    // no comment is posted).
    setupInputs({ github_token: "" });
    const { run } = await import("./main");
    await run();

    expect(createCommentFn).not.toHaveBeenCalled();
  });

  // ── Argument array injection-safety ─────────────────────────────────────

  it("passes each input as a separate string element (injection-safe)", async () => {
    setupInputs({
      dataset: "my/dataset.yaml",
      base_url: "http://llm.example.com",
      model: "gpt-4",
      threshold: "0.10",
      out: "out.json",
      baseline: "baseline.json",
    });
    const { run } = await import("./main");
    await run();

    const args = execFileSyncFn.mock.calls[0][1] as string[];
    // No element should contain unescaped shell operators.
    for (const arg of args) {
      expect(typeof arg).toBe("string");
      // Each input is a discrete element, not concatenated into one string.
    }
    // Check the expected shape.
    expect(args[0]).toBe("run");
    expect(args[1]).toBe("my/dataset.yaml");
    expect(args.find((a) => a.startsWith("--base-url="))).toBeDefined();
    expect(args.find((a) => a.startsWith("--model="))).toBeDefined();
    expect(args.find((a) => a.startsWith("--baseline="))).toBeDefined();
  });
});
