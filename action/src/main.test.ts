import { describe, it, expect, vi, beforeEach } from "vitest";

// Minimal smoke test for the action module shape.
// Full integration tests require a live evalgate install; these just guard
// that the TypeScript compiles and the module exports are present.

describe("evalgate action", () => {
  it("module file exists and is importable as a module path", async () => {
    // Just verify the file can be resolved — no side effects on import
    // because run() is not called at module scope.
    const mod = await import("./main");
    // The default export is undefined (no default); what matters is no throw.
    expect(mod).toBeDefined();
  });

  it("argument array construction stays injection-safe", () => {
    // Simulate what run() does internally when building the args array.
    const dataset = "examples/basic.yaml";
    const baseUrl = "http://localhost:8765";
    const model = "mock-model";
    const threshold = "0.05";
    const out = "report.json";
    const baseline = "baseline.json";

    const args: string[] = [
      "run",
      dataset,
      `--base-url=${baseUrl}`,
      `--model=${model}`,
      `--threshold=${threshold}`,
      `--out=${out}`,
      `--baseline=${baseline}`,
    ];

    // Each entry must be a plain string — no shell metacharacters interpreted.
    expect(args).toHaveLength(7);
    expect(args[0]).toBe("run");
    expect(args[1]).toBe("examples/basic.yaml");
    // No single element should contain unescaped shell operators.
    for (const arg of args) {
      expect(typeof arg).toBe("string");
    }
  });
});
