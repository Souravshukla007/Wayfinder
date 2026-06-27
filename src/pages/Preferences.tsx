import { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Check, Plus, Sparkles, X } from "lucide-react";

type Topic = { key: string; label: string; color: string; weight: number };

const INITIAL_TOPICS: Topic[] = [
  { key: "photography", label: "Photography", color: "hsl(340 75% 58%)", weight: 0.9 },
  { key: "food", label: "Local Food", color: "hsl(20 85% 60%)", weight: 0.75 },
  { key: "anime", label: "Anime", color: "hsl(280 65% 60%)", weight: 0.55 },
  { key: "nature", label: "Nature", color: "hsl(150 55% 45%)", weight: 0.7 },
  { key: "luxury", label: "Luxury", color: "hsl(45 90% 55%)", weight: 0.3 },
  { key: "adventure", label: "Adventure", color: "hsl(15 80% 55%)", weight: 0.6 },
  { key: "history", label: "History", color: "hsl(210 60% 50%)", weight: 0.65 },
  { key: "nightlife", label: "Nightlife", color: "hsl(330 70% 55%)", weight: 0.2 },
];

function SliderRow({
  topic,
  onChange,
}: {
  topic: Topic;
  onChange: (w: number) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const pct = Math.round(topic.weight * 100);
  return (
    <div className="py-2.5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">{topic.label}</span>
        <span
          className="text-xs font-semibold tabular-nums"
          style={{ color: topic.color }}
        >
          {topic.weight.toFixed(2)}
        </span>
      </div>
      <div className="relative h-3">
        {/* track */}
        <div className="absolute inset-y-1/2 -translate-y-1/2 left-0 right-0 h-1.5 bg-muted rounded-full" />
        {/* fill */}
        <div
          className="absolute inset-y-1/2 -translate-y-1/2 left-0 h-1.5 rounded-full transition-[width] duration-75"
          style={{ width: `${pct}%`, background: topic.color }}
        />
        {/* thumb */}
        <motion.div
          aria-hidden
          className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-card border-2 shadow-sm pointer-events-none"
          style={{ left: `${pct}%`, borderColor: topic.color }}
          animate={{ scale: dragging ? 1.15 : 1 }}
          transition={{ type: "spring", stiffness: 400, damping: 18 }}
        />
        {/* native input on top, transparent */}
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={topic.weight}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          onPointerDown={() => setDragging(true)}
          onPointerUp={() => setDragging(false)}
          onPointerLeave={() => setDragging(false)}
          aria-label={`${topic.label} weight`}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
      </div>
    </div>
  );
}

function PillList({
  variant,
  items,
  setItems,
  placeholder,
}: {
  variant: "loved" | "skip";
  items: string[];
  setItems: (next: string[]) => void;
  placeholder: string;
}) {
  const [input, setInput] = useState("");
  const add = () => {
    const v = input.trim();
    if (!v || items.includes(v)) return;
    setItems([...items, v]);
    setInput("");
  };
  const remove = (v: string) => setItems(items.filter((x) => x !== v));

  const pillClass =
    variant === "loved"
      ? "bg-[hsl(var(--success))]/15 text-[hsl(var(--success))]"
      : "bg-muted text-muted-foreground";

  return (
    <div className="bg-card rounded-3xl p-5 shadow-sm">
      <div className="text-sm font-semibold mb-3">
        {variant === "loved" ? "Places I loved" : "Places to skip"}
      </div>

      <div className="flex flex-wrap gap-2 min-h-[36px]">
        <AnimatePresence initial={false}>
          {items.map((it) => (
            <motion.span
              key={it}
              layout
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 22, duration: 0.2 }}
              className={`inline-flex items-center gap-1.5 pl-3 pr-1.5 h-8 rounded-full text-xs font-semibold ${pillClass}`}
            >
              {it}
              <button
                type="button"
                onClick={() => remove(it)}
                aria-label={`Remove ${it}`}
                className="w-5 h-5 rounded-full hover:bg-foreground/10 flex items-center justify-center"
              >
                <X className="w-3 h-3" />
              </button>
            </motion.span>
          ))}
        </AnimatePresence>
      </div>

      <div className="mt-4 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              add();
            }
          }}
          placeholder={placeholder}
          className="flex-1 h-9 px-4 rounded-full bg-muted text-sm outline-none placeholder:text-muted-foreground/70 focus:bg-background focus:ring-2 focus:ring-primary/30 transition"
        />
        <button
          type="button"
          onClick={add}
          aria-label="Add"
          className="w-9 h-9 rounded-full bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

export default function Preferences() {
  const [topics, setTopics] = useState<Topic[]>(INITIAL_TOPICS);
  const [loved, setLoved] = useState<string[]>(["Kyoto", "Lisbon", "Reykjavík"]);
  const [skip, setSkip] = useState<string[]>(["Cancún", "Las Vegas"]);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");

  const setWeight = (key: string, w: number) =>
    setTopics((ts) => ts.map((t) => (t.key === key ? { ...t, weight: w } : t)));

  const onSave = () => {
    if (saveState !== "idle") return;
    setSaveState("saving");
    setTimeout(() => setSaveState("saved"), 400);
    setTimeout(() => setSaveState("idle"), 1600);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="h-16" aria-hidden />

      <main className="px-6 py-10 md:py-16">
        <div className="max-w-[600px] mx-auto">
          <div className="text-center mb-10">
            <h1 className="font-display font-bold text-4xl md:text-5xl tracking-[-0.02em]">
              My preferences
            </h1>
            <p className="mt-3 text-muted-foreground">
              Tune what matters to you. Every plan adapts to this vector.
            </p>
          </div>

          {/* Preference vector */}
          <div className="bg-card rounded-3xl p-6 md:p-7 shadow-sm">
            <div className="text-sm font-semibold mb-2">Interest vector</div>
            <p className="text-xs text-muted-foreground mb-4">
              Drag to weight each topic from 0.00 to 1.00.
            </p>
            <div className="divide-y divide-border/50">
              {topics.map((t) => (
                <SliderRow
                  key={t.key}
                  topic={t}
                  onChange={(w) => setWeight(t.key, w)}
                />
              ))}
            </div>
          </div>

          {/* Pill lists */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6">
            <PillList
              variant="loved"
              items={loved}
              setItems={setLoved}
              placeholder="Add a place…"
            />
            <PillList
              variant="skip"
              items={skip}
              setItems={setSkip}
              placeholder="Add a place…"
            />
          </div>

          {/* Save */}
          <motion.button
            type="button"
            onClick={onSave}
            disabled={saveState !== "idle"}
            animate={{
              backgroundColor:
                saveState === "saved"
                  ? "hsl(var(--success))"
                  : "hsl(var(--primary))",
            }}
            transition={{ duration: 0.25 }}
            className="mt-8 w-full h-14 rounded-full text-primary-foreground font-semibold text-base flex items-center justify-center gap-2 shadow-sm disabled:opacity-90"
          >
            <AnimatePresence mode="wait">
              <motion.span
                key={saveState}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
                className="inline-flex items-center gap-2"
              >
                {saveState === "saved" ? (
                  <>
                    Saved <Check className="w-4 h-4" />
                  </>
                ) : saveState === "saving" ? (
                  "Saving…"
                ) : (
                  "Save preferences"
                )}
              </motion.span>
            </AnimatePresence>
          </motion.button>
        </div>
      </main>
    </div>
  );
}
