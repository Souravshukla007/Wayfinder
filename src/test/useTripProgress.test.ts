import { describe, it, expect } from "vitest";
import { stepFromCompletedStages } from "@/hooks/useTripProgress";

// `stepFromCompletedStages` translates the count of completed backend pipeline
// stages into an index over the visual progress checklist. These tests pin its
// monotonic, clamped behavior — the contract the progress UI relies on.
describe("stepFromCompletedStages", () => {
  it("returns 0 when nothing has completed", () => {
    expect(stepFromCompletedStages(0, 8, 7)).toBe(0);
  });

  it("returns the full step count when every stage has completed", () => {
    expect(stepFromCompletedStages(7, 8, 7)).toBe(8);
  });

  it("maps partial progress proportionally and rounded", () => {
    // 4/7 of the stages done over an 8-item checklist => round(0.571 * 8) = 5.
    expect(stepFromCompletedStages(4, 8, 7)).toBe(5);
  });

  it("never exceeds totalSteps even with extra completions", () => {
    expect(stepFromCompletedStages(99, 8, 7)).toBe(8);
  });

  it("clamps negative completion counts to 0", () => {
    expect(stepFromCompletedStages(-3, 8, 7)).toBe(0);
  });

  it("is monotonic non-decreasing across the stage range", () => {
    let prev = -1;
    for (let done = 0; done <= 7; done += 1) {
      const step = stepFromCompletedStages(done, 8, 7);
      expect(step).toBeGreaterThanOrEqual(prev);
      prev = step;
    }
  });

  it("guards against zero/negative totals", () => {
    expect(stepFromCompletedStages(3, 0, 7)).toBe(0);
    expect(stepFromCompletedStages(3, 8, 0)).toBe(0);
  });
});
