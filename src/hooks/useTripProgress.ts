// Drives the trip-planning progress UI from the live WebSocket stage stream.
//
// `begin()` starts a real plan (`POST /api/trips/plan`) and subscribes to
// `WS /api/trips/{tripId}/progress`. Each completed backend stage advances the
// UI proportionally across the visual checklist. If planning or the stream
// fails, the hook falls back to a time-based progression so the prototype never
// hangs on the progress screen without a live backend.
//
// Requirement 13.1: the UI is driven by live agent-progress stage events.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  PROGRESS_STAGES,
  planTrip,
  subscribeToProgress,
  type ProgressEvent,
} from "@/lib/mockApi";

export type TripProgressPhase = "idle" | "running" | "done" | "error";

/**
 * Map the count of completed backend stages to an index into a UI checklist of
 * `totalSteps` items. Pure and monotonic: the returned index never exceeds
 * `totalSteps` and grows with `completedStages`. Exported for unit testing.
 */
export function stepFromCompletedStages(
  completedStages: number,
  totalSteps: number,
  totalStages: number = PROGRESS_STAGES.length,
): number {
  if (totalSteps <= 0 || totalStages <= 0) return 0;
  const clamped = Math.max(0, Math.min(completedStages, totalStages));
  const fraction = clamped / totalStages;
  return Math.min(totalSteps, Math.round(fraction * totalSteps));
}

type BeginInput = { prompt: string; interests?: string[] };

export type UseTripProgress = {
  phase: TripProgressPhase;
  /** Index into the visual checklist of the currently active step. */
  activeIdx: number;
  /** Overall completion percentage (0–100) for the progress bar. */
  progress: number;
  /** The most recent live stage message, if any. */
  message: string | null;
  tripId: string | null;
  /** Start a real planning run and subscribe to its progress stream. */
  begin: (input: BeginInput) => void;
  /** Tear down any stream/timer and return to the idle state. */
  reset: () => void;
};

const FALLBACK_STEP_MS = 700;

export function useTripProgress(totalSteps: number): UseTripProgress {
  const [phase, setPhase] = useState<TripProgressPhase>("idle");
  const [activeIdx, setActiveIdx] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [tripId, setTripId] = useState<string | null>(null);

  const unsubscribeRef = useRef<(() => void) | null>(null);
  const fallbackTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const completedStagesRef = useRef(0);

  const clearTimers = useCallback(() => {
    unsubscribeRef.current?.();
    unsubscribeRef.current = null;
    if (fallbackTimerRef.current != null) {
      clearInterval(fallbackTimerRef.current);
      fallbackTimerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    clearTimers();
    completedStagesRef.current = 0;
    setPhase("idle");
    setActiveIdx(0);
    setMessage(null);
    setTripId(null);
  }, [clearTimers]);

  // Time-based progression used when no live backend stream is available.
  const startFallback = useCallback(() => {
    clearTimers();
    fallbackTimerRef.current = setInterval(() => {
      setActiveIdx((i) => {
        if (i >= totalSteps) {
          if (fallbackTimerRef.current != null) {
            clearInterval(fallbackTimerRef.current);
            fallbackTimerRef.current = null;
          }
          setPhase("done");
          return totalSteps;
        }
        return i + 1;
      });
    }, FALLBACK_STEP_MS);
  }, [clearTimers, totalSteps]);

  const begin = useCallback(
    (input: BeginInput) => {
      clearTimers();
      completedStagesRef.current = 0;
      setActiveIdx(0);
      setMessage(null);
      setPhase("running");

      void (async () => {
        let id: string;
        try {
          const res = await planTrip(input);
          id = res.tripId;
          setTripId(id);
        } catch {
          // Backend unavailable / unauthenticated — keep the UX flowing.
          startFallback();
          return;
        }

        unsubscribeRef.current = subscribeToProgress(id, {
          onEvent: (event: ProgressEvent) => {
            setMessage(event.message);
            if (event.phase === "complete") {
              completedStagesRef.current += 1;
              setActiveIdx(
                stepFromCompletedStages(completedStagesRef.current, totalSteps),
              );
            }
          },
          onComplete: () => {
            setActiveIdx(totalSteps);
            setPhase("done");
          },
          onError: () => {
            // Stream rejected/failed after planning succeeded — degrade to the
            // time-based progression rather than stalling.
            startFallback();
          },
        });
      })();
    },
    [clearTimers, startFallback, totalSteps],
  );

  // Clean up any open socket/timer when the consumer unmounts.
  useEffect(() => clearTimers, [clearTimers]);

  const progress = totalSteps > 0 ? Math.min(100, (activeIdx / totalSteps) * 100) : 0;

  return { phase, activeIdx, progress, message, tripId, begin, reset };
}
