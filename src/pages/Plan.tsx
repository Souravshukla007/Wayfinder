import { useState, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Loader2,
  Check,
  Brain,
  Plane,
  Hotel,
  CloudSun,
  PartyPopper,
  Map as MapIcon,
  Sparkles,
  ClipboardList,
  Star,
  ArrowLeft,
  ChevronDown,
  Camera,
  Users,
  Wallet,
  MapPin,
  Info,
} from "lucide-react";
import { MOCK_RESULTS, BAR_COLORS, getResults, type Result, type BreakdownRow, type DecisionTrace, type DayPlan, type FlightOpt, type HotelOpt, type EventOpt } from "@/lib/mockApi";
import { useTripProgress } from "@/hooks/useTripProgress";
import { toast } from "sonner";

const PLACEHOLDERS = [
  "7-day Japan trip in October under ₹2 lakh, I like anime and photography...",
  "5 days in South Korea, vegetarian, less crowded places...",
  "Weekend in Rajasthan, heritage hotels, under ₹40,000...",
];

const INTEREST_TAGS = [
  "Photography",
  "Local Food",
  "Less Crowded",
  "Anime",
  "Nature",
  "Adventure",
];

type Step = "input" | "progress" | "results";

type Task = {
  key: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
};

const TASKS: Task[] = [
  { key: "understand", label: "Understanding your request", icon: Sparkles },
  { key: "flights", label: "Searching flights", icon: Plane },
  { key: "hotels", label: "Finding hotels", icon: Hotel },
  { key: "weather", label: "Checking weather for October", icon: CloudSun },
  { key: "events", label: "Scanning local events & festivals", icon: PartyPopper },
  { key: "routes", label: "Optimizing routes", icon: MapIcon },
  { key: "scoring", label: "Scoring destinations", icon: Brain },
  { key: "itinerary", label: "Building your itinerary", icon: ClipboardList },
];

function AnimatedPlaceholder({ active }: { active: boolean }) {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setIdx((i) => (i + 1) % PLACEHOLDERS.length), 3000);
    return () => clearInterval(t);
  }, [active]);
  return (
    <div className="absolute inset-0 pointer-events-none px-5 pt-5 text-muted-foreground">
      <AnimatePresence mode="wait">
        <motion.span
          key={idx}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.4 }}
          className="block text-base leading-relaxed"
        >
          {PLACEHOLDERS[idx]}
        </motion.span>
      </AnimatePresence>
    </div>
  );
}

function InputPanel({
  value,
  setValue,
  selected,
  toggleTag,
  loading,
  onSubmit,
}: {
  value: string;
  setValue: (v: string) => void;
  selected: string[];
  toggleTag: (t: string) => void;
  loading: boolean;
  onSubmit: () => void;
}) {
  const taRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 280) + "px";
  }, [value]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="w-full"
    >
      <div className="text-center mb-10">
        <h1 className="font-display font-bold text-4xl md:text-5xl tracking-[-0.02em]">
          Where to next?
        </h1>
        <p className="mt-3 text-muted-foreground">
          Describe your trip in plain words. Wayfinder handles the rest.
        </p>
      </div>

      <div className="relative bg-card rounded-3xl shadow-sm p-1">
        <div className="relative">
          {value.length === 0 && <AnimatedPlaceholder active={!loading} />}
          <textarea
            ref={taRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            rows={3}
            className="relative w-full resize-none bg-transparent px-5 pt-5 pb-3 text-base leading-relaxed outline-none placeholder:text-transparent min-h-[120px]"
            placeholder="Describe your trip..."
          />
        </div>
      </div>

      <div className="mt-6">
        <div className="text-xs font-semibold text-muted-foreground mb-3 px-1">
          Pick interests to refine
        </div>
        <div className="flex flex-wrap gap-2">
          {INTEREST_TAGS.map((tag) => {
            const isOn = selected.includes(tag);
            return (
              <motion.button
                key={tag}
                type="button"
                onClick={() => toggleTag(tag)}
                animate={{ scale: isOn ? 1.04 : 1 }}
                whileTap={{ scale: 0.96 }}
                transition={{ type: "spring", stiffness: 400, damping: 22 }}
                className={`px-4 h-9 rounded-full text-sm font-medium transition-colors ${
                  isOn
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/70"
                }`}
              >
                {tag}
              </motion.button>
            );
          })}
        </div>
      </div>

      <div className="mt-8 flex justify-center">
        <motion.button
          type="button"
          onClick={onSubmit}
          disabled={loading || value.trim().length === 0}
          animate={{ width: loading ? 56 : "100%", borderRadius: 9999 }}
          transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          className="h-14 bg-primary text-primary-foreground font-semibold text-base flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm overflow-hidden"
          style={{ maxWidth: "100%" }}
        >
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.span
                key="loader"
                initial={{ opacity: 0, scale: 0.6 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.6 }}
                transition={{ duration: 0.2 }}
              >
                <Loader2 className="w-5 h-5 animate-spin" />
              </motion.span>
            ) : (
              <motion.span
                key="label"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="inline-flex items-center gap-2 whitespace-nowrap"
              >
                Plan my trip <ArrowRight className="w-4 h-4" />
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      </div>
    </motion.div>
  );
}

function ProgressPanel({
  progress,
  activeIdx,
}: {
  progress: number;
  activeIdx: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="w-full"
    >
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold mb-4">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          Working on it
        </div>
        <h2 className="font-display font-bold text-3xl md:text-4xl tracking-[-0.02em]">
          Reasoning through your trip
        </h2>
      </div>

      <div className="bg-card rounded-3xl p-6 md:p-8 shadow-sm">
        <div className="h-1 w-full bg-muted rounded-full overflow-hidden mb-6">
          <motion.div
            className="h-full bg-primary rounded-full"
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        </div>

        <ul className="space-y-1">
          {TASKS.map((task, i) => {
            const Icon = task.icon;
            const status: "done" | "live" | "pending" =
              i < activeIdx ? "done" : i === activeIdx ? "live" : "pending";
            return (
              <motion.li
                key={task.key}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.2, duration: 0.35 }}
                className="flex items-center gap-3 py-2.5"
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    status === "pending" ? "text-muted-foreground/50" : "text-foreground"
                  }`}
                />
                <span
                  className={`flex-1 text-sm ${
                    status === "done"
                      ? "text-muted-foreground"
                      : status === "live"
                        ? "font-semibold text-foreground"
                        : "text-muted-foreground/60"
                  }`}
                >
                  {task.label}
                  {status === "live" && "..."}
                </span>
                <StatusIndicator status={status} />
              </motion.li>
            );
          })}
        </ul>
      </div>
    </motion.div>
  );
}

function StatusIndicator({ status }: { status: "done" | "live" | "pending" }) {
  if (status === "done") {
    return (
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 400, damping: 18 }}
        className="w-5 h-5 rounded-full bg-[hsl(var(--success))] flex items-center justify-center"
      >
        <Check className="w-3 h-3 text-[hsl(var(--success-foreground))]" />
      </motion.div>
    );
  }
  if (status === "live") {
    return (
      <span className="relative flex w-3 h-3">
        <span className="absolute inset-0 rounded-full bg-primary/50 animate-ping" />
        <span className="relative inline-flex w-3 h-3 rounded-full bg-primary" />
      </span>
    );
  }
  return <span className="w-3 h-3 rounded-full border-2 border-muted-foreground/30" />;
}

// Result types, BAR_COLORS, and MOCK_RESULTS now live in @/lib/mockApi (single source of truth).

function FeatureBars({ rows, runKey }: { rows: BreakdownRow[]; runKey: string }) {
  const total = rows.reduce((a, b) => a + b.val, 0);
  return (
    <div>
      <div className="space-y-3">
        {rows.map((r, i) => (
          <div key={r.label} className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground w-28 shrink-0">{r.label}</span>
            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
              <motion.div
                key={runKey + r.label}
                initial={{ width: 0 }}
                animate={{ width: `${r.val * 4}%` }}
                transition={{ duration: 0.6, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
                className="h-full rounded-full"
                style={{ background: r.color }}
              />
            </div>
            <span className="text-xs font-semibold w-10 text-right tabular-nums" style={{ color: rows[i].color }}>
              +{r.val}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-baseline justify-end gap-2 border-t border-border pt-3">
        <span className="text-xs text-muted-foreground uppercase tracking-wide">Total</span>
        <span className="font-display font-bold text-lg tabular-nums">{total}</span>
        <span className="text-xs text-muted-foreground">/ 100</span>
      </div>
    </div>
  );
}

// Renders the decision-trace "Why X Won" breakdown for the chosen destination:
// each per-feature contribution from the backend decision trace, shown alongside
// the trace's final score (Requirement 8.6).
export function DecisionTraceBreakdown({ trace, runKey }: { trace: DecisionTrace; runKey: string }) {
  const factors = trace.factors;
  const maxContribution = factors.reduce((m, f) => Math.max(m, f.contribution), 0);
  return (
    <div>
      <div className="space-y-3">
        {factors.map((f, i) => {
          const color = BAR_COLORS[i % BAR_COLORS.length];
          // Scale bar width relative to the largest contribution so the strongest
          // factor fills the track; guard against a zero/negative max.
          const widthPct = maxContribution > 0 ? (f.contribution / maxContribution) * 100 : 0;
          return (
            <div key={f.feature} className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground w-28 shrink-0">{f.feature}</span>
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  key={runKey + f.feature}
                  initial={{ width: 0 }}
                  animate={{ width: `${widthPct}%` }}
                  transition={{ duration: 0.6, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
                  className="h-full rounded-full"
                  style={{ background: color }}
                />
              </div>
              <span
                className="text-xs font-semibold w-10 text-right tabular-nums"
                style={{ color }}
              >
                +{Number.isInteger(f.contribution) ? f.contribution : f.contribution.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-4 flex items-baseline justify-end gap-2 border-t border-border pt-3">
        <span className="text-xs text-muted-foreground uppercase tracking-wide">Total</span>
        <span className="font-display font-bold text-lg tabular-nums">
          {Number.isInteger(trace.score) ? trace.score : trace.score.toFixed(1)}
        </span>
        <span className="text-xs text-muted-foreground">/ 100</span>
      </div>
    </div>
  );
}

// Fallback shown when the chosen destination has no decision trace: a clear
// message that the per-feature contribution details are unavailable for that
// destination (Requirement 8.7).
export function NoTraceMessage({ destinationName }: { destinationName: string }) {
  return (
    <div
      role="status"
      className="flex items-start gap-3 rounded-2xl bg-muted/60 px-4 py-4 text-sm text-muted-foreground"
    >
      <Info className="w-4 h-4 shrink-0 mt-0.5" aria-hidden />
      <p className="leading-relaxed">
        Contribution details are unavailable for {destinationName}.
      </p>
    </div>
  );
}

// Renders the whole-ranking narration produced by the backend: deterministic
// ranking facts followed by live LLM prose (when an LLM vendor is configured).
// Explanatory only — it never overrides the scores or order shown above.
export function NarrationCard({ text }: { text: string }) {
  return (
    <div className="mt-6 rounded-2xl border border-primary/15 bg-primary/5 px-4 py-4">
      <div className="flex items-center gap-2 text-xs font-semibold text-primary mb-2">
        <Sparkles className="w-3.5 h-3.5" aria-hidden />
        AI explanation
      </div>
      <p className="text-sm leading-relaxed text-foreground/80 whitespace-pre-line">
        {text}
      </p>
    </div>
  );
}

type TabKey = "itinerary" | "flights" | "hotels" | "events";

function DetailTabs({ result }: { result: Result }) {
  const [tab, setTab] = useState<TabKey>("itinerary");
  const tabs: { key: TabKey; label: string }[] = [
    { key: "itinerary", label: "Itinerary" },
    { key: "flights", label: "Flights" },
    { key: "hotels", label: "Hotels" },
    { key: "events", label: "Events" },
  ];

  return (
    <div>
      <div className="flex gap-1 p-1 bg-muted rounded-full mb-5 w-fit">
        {tabs.map((t) => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`relative px-4 h-8 rounded-full text-xs font-semibold transition-colors ${
                active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="detail-tab-pill"
                  className="absolute inset-0 bg-primary rounded-full"
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}
              <span className="relative">{t.label}</span>
            </button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={tab + result.grade}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {tab === "itinerary" && <ItineraryView days={result.itinerary} />}
          {tab === "flights" && <FlightsView flights={result.flights} />}
          {tab === "hotels" && <HotelsView hotels={result.hotels} />}
          {tab === "events" && <EventsView events={result.events} />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

function ItineraryView({ days }: { days: DayPlan[] }) {
  const [open, setOpen] = useState<number | null>(0);
  return (
    <div className="space-y-2">
      {days.map((d, i) => {
        const isOpen = open === i;
        return (
          <div key={i} className="bg-background rounded-2xl overflow-hidden">
            <button
              onClick={() => setOpen(isOpen ? null : i)}
              className="w-full flex items-center gap-3 px-4 py-3 text-left"
            >
              <span className="text-xs font-semibold text-primary w-12 shrink-0">{d.day}</span>
              <span className="flex-1 text-sm font-medium">{d.title}</span>
              <motion.span animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.25 }}>
                <ChevronDown className="w-4 h-4 text-muted-foreground" />
              </motion.span>
            </button>
            <div
              className="grid transition-[grid-template-rows] duration-300 ease-out"
              style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}
            >
              <div className="overflow-hidden">
                <ul className="px-4 pb-4 pt-1 space-y-1.5 pl-[68px]">
                  {d.items.map((it, k) => (
                    <li key={k} className="text-sm text-muted-foreground flex gap-2">
                      <span className="text-primary mt-1.5 w-1 h-1 rounded-full bg-primary shrink-0" />
                      {it}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function FlightsView({ flights }: { flights: FlightOpt[] }) {
  return (
    <div className="space-y-2">
      {flights.map((f, i) => (
        <div key={i} className="bg-background rounded-2xl px-4 py-3 flex items-center gap-4">
          <div className="w-9 h-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center shrink-0">
            <Plane className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold">{f.airline}</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-primary/10 text-primary font-semibold">{f.tag}</span>
            </div>
            <div className="text-xs text-muted-foreground">{f.duration}</div>
          </div>
          <div className="font-display font-bold text-sm tabular-nums">{f.price}</div>
        </div>
      ))}
    </div>
  );
}

function HotelsView({ hotels }: { hotels: HotelOpt[] }) {
  return (
    <div className="space-y-2">
      {hotels.map((h, i) => (
        <div key={i} className="bg-background rounded-2xl px-4 py-3 flex items-center gap-4">
          <div className="w-9 h-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center shrink-0">
            <Hotel className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold">{h.name}</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-primary/10 text-primary font-semibold">{h.tag}</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="flex gap-0.5">
                {Array.from({ length: 5 }).map((_, k) => (
                  <Star key={k} className={`w-3 h-3 ${k < h.rating ? "fill-primary text-primary" : "text-muted"}`} />
                ))}
              </span>
              <span>·</span>
              <span>{h.distance}</span>
            </div>
          </div>
          <div className="font-display font-bold text-sm tabular-nums">{h.price}</div>
        </div>
      ))}
    </div>
  );
}

function EventsView({ events }: { events: EventOpt[] }) {
  return (
    <ol className="relative border-l-2 border-dashed border-border ml-3 space-y-4">
      {events.map((e, i) => (
        <li key={i} className="pl-5 relative">
          <span className="absolute -left-[7px] top-1.5 w-3 h-3 rounded-full bg-primary" />
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold">{e.name}</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-secondary text-secondary-foreground font-semibold">{e.type}</span>
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">{e.date}</div>
        </li>
      ))}
    </ol>
  );
}

function ResultCard({
  r,
  i,
  selected,
  onSelect,
}: {
  r: Result;
  i: number;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <motion.button
      type="button"
      onClick={onSelect}
      initial={{ opacity: 0, x: -32 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: i * 0.12, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -3 }}
      className={`w-full text-left bg-card rounded-3xl p-5 shadow-sm transition-shadow hover:shadow-lg relative ${
        selected ? "ring-2 ring-primary" : ""
      }`}
    >
      {selected && (
        <span className="absolute left-0 top-6 bottom-6 w-1 rounded-r-full bg-primary" aria-hidden />
      )}
      <div className="flex items-start gap-3">
        <span className="shrink-0 w-9 h-9 rounded-xl bg-primary text-primary-foreground font-display font-bold flex items-center justify-center">
          {r.grade}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h3 className="font-display font-bold text-base leading-tight">
                {r.name} <span className="ml-1">{r.flag}</span>
              </h3>
              <div className="text-xs text-muted-foreground mt-0.5">{r.country}</div>
            </div>
            <div className="text-right">
              <div className="font-display font-bold text-xl text-primary tabular-nums leading-none">
                {r.score}
              </div>
              <div className="text-[9px] uppercase tracking-wide text-muted-foreground">score</div>
            </div>
          </div>

          <div className="flex items-center gap-3 mt-3 text-[11px] text-muted-foreground">
            {r.micro.map((m, k) => {
              const Icon = m.icon;
              return (
                <span key={k} className="inline-flex items-center gap-1">
                  <Icon className="w-3 h-3" />
                  {m.label}
                </span>
              );
            })}
          </div>

          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">{r.teaser}</p>

          {r.badge && (
            <span className="inline-flex items-center gap-1 mt-3 px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-semibold">
              <Star className="w-2.5 h-2.5 fill-primary" /> {r.badge}
            </span>
          )}
        </div>
      </div>
    </motion.button>
  );
}

function ResultsPanel({ onReset, tripId }: { onReset: () => void; tripId: string | null }) {
  // Live results (with the whole-ranking narration) replace the mock seed once
  // the backend responds; on any failure the mock data keeps the UI populated.
  const [results, setResults] = useState<Result[]>(MOCK_RESULTS);
  const [narration, setNarration] = useState<string>("");
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [booking, setBooking] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!tripId) return;
    let cancelled = false;
    void (async () => {
      try {
        const bundle = await getResults(tripId);
        if (cancelled || bundle.results.length === 0) return;
        setResults(bundle.results);
        setNarration(bundle.narration);
        setSelectedIdx(0);
      } catch {
        // Backend unavailable — keep the mock fallback already in state.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tripId]);

  const selected = results[selectedIdx] ?? results[0];

  // "Save" — the trip is already persisted during planning, so this confirms
  // it and takes the user to their trip history.
  const handleSave = () => {
    toast.success(`"${selected.name}" saved to your trips`);
    setTimeout(() => navigate("/trips"), 700);
  };

  // "Book this plan" — in-app booking/payment isn't part of this product (and
  // flights/hotels are sourced from providers, not a booking engine), so we
  // route the traveler to live booking search for the chosen destination
  // rather than pretending to complete a booking.
  const handleBook = () => {
    setBooking(true);
    const city = selected.name;
    const flightsUrl = `https://www.google.com/travel/flights?q=${encodeURIComponent(
      `flights to ${city}`,
    )}`;
    const staysUrl = `https://www.google.com/travel/search?q=${encodeURIComponent(
      `hotels in ${city}`,
    )}`;
    window.open(flightsUrl, "_blank", "noopener,noreferrer");
    window.open(staysUrl, "_blank", "noopener,noreferrer");
    toast.success(`Opening flight & stay booking options for ${city}`);
    setTimeout(() => setBooking(false), 1200);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="w-full"
    >
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[hsl(var(--success))]/15 text-[hsl(var(--success))] text-xs font-semibold mb-4">
          <Check className="w-3.5 h-3.5" />
          Plan ready
        </div>
        <h2 className="font-display font-bold text-3xl md:text-4xl tracking-[-0.02em]">
          Three ranked options
        </h2>
        <p className="mt-3 text-muted-foreground">
          Tap any card to see the full reasoning and itinerary.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)] gap-6">
        {/* Left column */}
        <div className="space-y-3">
          {results.map((r, i) => (
            <ResultCard
              key={r.grade}
              r={r}
              i={i}
              selected={i === selectedIdx}
              onSelect={() => setSelectedIdx(i)}
            />
          ))}
        </div>

        {/* Right column */}
        <AnimatePresence mode="wait">
          <motion.div
            key={selected.grade}
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="bg-card rounded-3xl p-5 md:p-7 shadow-sm"
          >
            <div className="rounded-2xl overflow-hidden mb-5">
              <img src={selected.image} alt={selected.name} className="w-full h-48 md:h-56 object-cover" />
            </div>

            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <div className="text-xs font-semibold text-primary mb-1">Why {selected.name.split(" ")[0]} won</div>
                <h3 className="font-display font-bold text-2xl tracking-[-0.01em]">
                  {selected.name} <span>{selected.flag}</span>
                </h3>
              </div>
              <div className="text-right">
                <div className="font-display font-bold text-3xl text-primary tabular-nums leading-none">
                  {selected.score}
                </div>
                <div className="text-[10px] uppercase tracking-wide text-muted-foreground">total score</div>
              </div>
            </div>

            <div className="mt-6">
              {selected.decisionTrace ? (
                <DecisionTraceBreakdown trace={selected.decisionTrace} runKey={selected.grade} />
              ) : (
                <NoTraceMessage destinationName={selected.name} />
              )}
            </div>

            {narration && <NarrationCard text={narration} />}

            <div className="mt-8">
              <DetailTabs result={selected} />
            </div>

            <div className="mt-6 flex gap-2">
              <Button className="font-semibold" onClick={handleBook} disabled={booking}>
                {booking ? "Opening…" : "Book this plan"} <ArrowRight className="w-4 h-4" />
              </Button>
              <Button variant="ghost" className="font-medium" onClick={handleSave}>Save</Button>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-10 text-center">
        <Button variant="ghost" onClick={onReset} className="font-medium">
          <ArrowLeft className="w-4 h-4" /> Plan another trip
        </Button>
      </div>
    </motion.div>
  );
}

export default function Plan() {
  const [step, setStep] = useState<Step>("input");
  const [value, setValue] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  // Live agent-progress, driven by the WebSocket stage stream (Requirement 13.1).
  const { activeIdx, progress: streamProgress, phase, begin, tripId, reset: resetProgress } =
    useTripProgress(TASKS.length);

  const toggleTag = (t: string) =>
    setSelected((s) => (s.includes(t) ? s.filter((x) => x !== t) : [...s, t]));

  const onSubmit = () => {
    if (!value.trim() || loading) return;
    setLoading(true);
    // Kick off the real plan + progress subscription, then reveal the panel.
    begin({ prompt: value, interests: selected.length ? selected : undefined });
    setStep("progress");
    setLoading(false);
  };

  // When the live stream (or fallback) reports completion, reveal the results.
  useEffect(() => {
    if (step !== "progress" || phase !== "done") return;
    const t = setTimeout(() => setStep("results"), 500);
    return () => clearTimeout(t);
  }, [step, phase]);

  const progress =
    step === "progress" ? streamProgress : step === "results" ? 100 : 0;

  const reset = () => {
    resetProgress();
    setStep("input");
    setValue("");
    setSelected([]);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Slim top progress bar */}
      <div className="fixed top-0 left-0 right-0 h-0.5 bg-transparent z-50">
        <motion.div
          className="h-full bg-primary"
          animate={{ width: `${progress}%`, opacity: step === "input" ? 0 : 1 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>

      <div className="h-16" aria-hidden />

      <main className="px-6 py-10 md:py-16">
        <div className={`mx-auto ${step === "results" ? "max-w-6xl" : "max-w-[680px]"}`}>
          <AnimatePresence mode="wait">
            {step === "input" && (
              <InputPanel
                key="input"
                value={value}
                setValue={setValue}
                selected={selected}
                toggleTag={toggleTag}
                loading={loading}
                onSubmit={onSubmit}
              />
            )}
            {step === "progress" && (
              <ProgressPanel
                key="progress"
                progress={progress}
                activeIdx={activeIdx}
              />
            )}
            {step === "results" && <ResultsPanel key="results" onReset={reset} tripId={tripId} />}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
