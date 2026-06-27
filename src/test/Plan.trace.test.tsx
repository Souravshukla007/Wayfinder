import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DecisionTraceBreakdown, NoTraceMessage } from "@/pages/Plan";
import type { DecisionTrace } from "@/lib/mockApi";

// These tests pin the two decision-transparency rendering paths the results
// panel chooses between for a chosen destination:
//   - a per-feature contribution breakdown when a Decision_Trace exists (Req 8.6)
//   - a "contribution details unavailable" fallback when it does not (Req 8.7)

describe("DecisionTraceBreakdown (Requirement 8.6)", () => {
  const trace: DecisionTrace = {
    destination: "Kyoto + Hakone loop",
    score: 92,
    factors: [
      { feature: "Photography", contribution: 22 },
      { feature: "Crowd Density", contribution: 18 },
      { feature: "Weather", contribution: 17 },
      { feature: "Food", contribution: 14 },
      { feature: "Travel Efficiency", contribution: 12 },
      { feature: "Budget Fit", contribution: 9 },
    ],
  };

  it("renders one labeled row per feature contribution from the trace", () => {
    render(<DecisionTraceBreakdown trace={trace} runKey="A" />);

    for (const factor of trace.factors) {
      // The feature label is shown for every contribution.
      expect(screen.getByText(factor.feature)).toBeInTheDocument();
      // The additive contribution value is shown as "+N".
      expect(screen.getByText(`+${factor.contribution}`)).toBeInTheDocument();
    }
  });

  it("renders the trace's final score alongside the contributions", () => {
    render(<DecisionTraceBreakdown trace={trace} runKey="A" />);

    expect(screen.getByText(String(trace.score))).toBeInTheDocument();
    expect(screen.getByText(/\/ 100/)).toBeInTheDocument();
  });

  it("formats a fractional contribution to one decimal place", () => {
    const fractionalTrace: DecisionTrace = {
      destination: "Osaka",
      score: 87.5,
      factors: [
        { feature: "Food", contribution: 21.4 },
        { feature: "Budget Fit", contribution: 66.1 },
      ],
    };
    render(<DecisionTraceBreakdown trace={fractionalTrace} runKey="B" />);

    expect(screen.getByText("+21.4")).toBeInTheDocument();
    expect(screen.getByText("87.5")).toBeInTheDocument();
  });
});

describe("NoTraceMessage (Requirement 8.7)", () => {
  it("renders the contribution-details-unavailable fallback for the destination", () => {
    render(<NoTraceMessage destinationName="Hokkaido nature" />);

    expect(
      screen.getByText(
        /Contribution details are unavailable for Hokkaido nature\./,
      ),
    ).toBeInTheDocument();
  });

  it("names the specific destination in the message", () => {
    render(<NoTraceMessage destinationName="Sapporo escape" />);

    expect(
      screen.getByText(/Contribution details are unavailable for Sapporo escape\./),
    ).toBeInTheDocument();
  });
});
