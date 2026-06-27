import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { motion, AnimatePresence, useInView } from "framer-motion";
import {
  ArrowRight,
  Star,
  Sparkles,
  Plane,
  Hotel,
  CloudSun,
  Calendar,
  Wallet,
  Map as MapIcon,
  MessageSquare,
  Workflow,
  ListChecks,
  Check,
} from "lucide-react";

// Destination imagery (remote, free) — four different countries
const DESTINATIONS = [
  { name: "Kyoto", country: "Japan", img: "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=600&q=80" },
  { name: "Rio de Janeiro", country: "Brazil", img: "https://images.unsplash.com/photo-1483729558449-99ef09a8c325?w=600&q=80" },
  { name: "Venice", country: "Italy", img: "https://images.unsplash.com/photo-1514890547357-a9ee288728e0?w=600&q=80" },
  { name: "Giza", country: "Egypt", img: "https://images.unsplash.com/photo-1568322445389-f64ac2515020?w=600&q=80" },
];

const ROTATING = ["smarter.", "deeper.", "freer."];

const SHAPE_COLORS = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF6BCB", "#FF9F43"];
type ShapeSeed = { left: string; top: string; size: number; type: "circle" | "square" | "triangle"; color: string; delay: number; dur: number };
const FLOAT_SHAPES: ShapeSeed[] = [
  { left: "8%", top: "18%", size: 12, type: "circle", color: SHAPE_COLORS[0], delay: 0, dur: 6 },
  { left: "14%", top: "72%", size: 10, type: "triangle", color: SHAPE_COLORS[2], delay: 1.2, dur: 7 },
  { left: "22%", top: "40%", size: 8, type: "square", color: SHAPE_COLORS[1], delay: 0.6, dur: 6.5 },
  { left: "32%", top: "82%", size: 7, type: "circle", color: SHAPE_COLORS[3], delay: 2, dur: 8 },
  { left: "48%", top: "12%", size: 9, type: "triangle", color: SHAPE_COLORS[4], delay: 1.5, dur: 7 },
  { left: "62%", top: "78%", size: 11, type: "square", color: SHAPE_COLORS[5], delay: 0.3, dur: 6 },
  { left: "74%", top: "30%", size: 8, type: "circle", color: SHAPE_COLORS[1], delay: 1.8, dur: 7.5 },
  { left: "82%", top: "68%", size: 10, type: "triangle", color: SHAPE_COLORS[0], delay: 0.9, dur: 6 },
  { left: "90%", top: "22%", size: 7, type: "square", color: SHAPE_COLORS[3], delay: 2.4, dur: 8 },
  { left: "55%", top: "55%", size: 6, type: "circle", color: SHAPE_COLORS[4], delay: 0.5, dur: 7 },
];

function FloatingShape({ s }: { s: ShapeSeed }) {
  const style: React.CSSProperties = {
    position: "absolute",
    left: s.left,
    top: s.top,
    width: s.size,
    height: s.type === "triangle" ? 0 : s.size,
    animation: `nomad-float ${s.dur}s ease-in-out ${s.delay}s infinite`,
  };
  if (s.type === "circle") return <div style={{ ...style, borderRadius: "50%", background: s.color }} />;
  if (s.type === "square") return <div style={{ ...style, borderRadius: 2, background: s.color, transform: "rotate(15deg)" }} />;
  return (
    <div
      style={{
        ...style,
        width: 0,
        borderLeft: `${s.size / 2}px solid transparent`,
        borderRight: `${s.size / 2}px solid transparent`,
        borderBottom: `${s.size * 0.9}px solid ${s.color}`,
      }}
    />
  );
}

// Card quadrants
const CARD_POSITIONS = [
  { className: "hidden md:block absolute left-[-20px] lg:left-[30px] top-[10px] w-[240px] lg:w-[280px]", rot: -4, delay: 0.2, from: { x: -60, y: -40 } },
  { className: "hidden md:block absolute right-[-20px] lg:right-[30px] top-[30px] w-[240px] lg:w-[280px]", rot: 4, delay: 0.35, from: { x: 60, y: -40 } },
  { className: "hidden md:block absolute left-[-20px] lg:left-[50px] bottom-[10px] w-[240px] lg:w-[280px]", rot: 3, delay: 0.5, from: { x: -60, y: 40 } },
  { className: "hidden md:block absolute right-[-20px] lg:right-[50px] bottom-[30px] w-[240px] lg:w-[280px]", rot: -4, delay: 0.65, from: { x: 60, y: 40 } },
];

function DestinationCard({ dest, idx, mouse }: { dest: typeof DESTINATIONS[number]; idx: number; mouse: { x: number; y: number } }) {
  const pos = CARD_POSITIONS[idx];
  // Each card drifts a tiny bit opposite/along mouse based on index
  const driftX = mouse.x * (idx % 2 === 0 ? -8 : 8);
  const driftY = mouse.y * (idx < 2 ? -8 : 8);
  return (
    <motion.div
      className={pos.className}
      initial={{ opacity: 0, scale: 0.4, x: pos.from.x, y: pos.from.y }}
      animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
      transition={{ type: "spring", stiffness: 220, damping: 22, delay: pos.delay }}
    >
      <motion.div
        className="rounded-2xl overflow-hidden shadow-xl bg-card"
        style={{ rotate: pos.rot }}
        animate={{ x: driftX, y: driftY }}
        transition={{ type: "spring", stiffness: 80, damping: 18 }}
      >
        <img src={dest.img} alt={dest.name} className="w-full h-[170px] object-cover" />
        <div className="px-3 py-2.5 flex items-center justify-between gap-2">
          <span className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
            {dest.name}
          </span>
          <span className="text-[10px] text-muted-foreground">{dest.country}</span>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ---------------- How it works ----------------
const AGENT_PILLS = [
  { label: "Flights", icon: Plane },
  { label: "Hotels", icon: Hotel },
  { label: "Weather", icon: CloudSun },
  { label: "Events", icon: Calendar },
  { label: "Budget", icon: Wallet },
  { label: "Routes", icon: MapIcon },
];

function StepCard({ index, children, tint }: { index: number; children: React.ReactNode; tint?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.2, ease: [0.22, 1, 0.36, 1] }}
      className={`relative flex-1 ${tint ?? "bg-card"} rounded-3xl p-6 lg:p-8 shadow-sm`}
    >
      {children}
    </motion.div>
  );
}

function AgentPipeline() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  return (
    <div ref={ref} className="flex flex-wrap gap-2 mt-4">
      {AGENT_PILLS.map((p, i) => {
        const Icon = p.icon;
        return (
          <motion.div
            key={p.label}
            initial={{ opacity: 0, y: 8, scale: 0.9 }}
            animate={inView ? { opacity: 1, y: 0, scale: 1 } : {}}
            transition={{ delay: i * 0.15, duration: 0.35 }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary text-secondary-foreground text-xs font-medium"
          >
            <Icon className="w-3.5 h-3.5" />
            {p.label}
          </motion.div>
        );
      })}
    </div>
  );
}

// ---------------- Bento ----------------
function BentoTile({ children, className = "", index, tint }: { children: React.ReactNode; className?: string; index: number; tint?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.2 });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.96 }}
      animate={inView ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className={`${tint ?? "bg-card"} rounded-3xl p-6 lg:p-8 shadow-sm ${className}`}
    >
      {children}
    </motion.div>
  );
}

function RankedScores() {
  const rows = [
    { label: "Kyoto", score: 92, color: "hsl(340 75% 58%)" },
    { label: "Osaka", score: 88, color: "hsl(45 90% 55%)" },
    { label: "Hokkaido", score: 84, color: "hsl(170 60% 45%)" },
  ];
  return (
    <div className="mt-6 space-y-3">
      {rows.map((r, i) => (
        <motion.div
          key={r.label}
          initial={{ opacity: 0, x: -10 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.1 }}
          className="flex items-center gap-3 bg-background rounded-2xl p-4"
        >
          <span className="text-sm font-semibold w-20">{String.fromCharCode(65 + i)} · {r.label}</span>
          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: r.color }}
              initial={{ width: 0 }}
              whileInView={{ width: `${r.score}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.2 + i * 0.1 }}
            />
          </div>
          <span className="text-sm font-bold tabular-nums w-8 text-right">{r.score}</span>
        </motion.div>
      ))}
    </div>
  );
}

function Explainability() {
  const rows = [
    { label: "Photography", val: 18 },
    { label: "Crowd", val: 15 },
    { label: "Weather", val: 14 },
    { label: "Food", val: 12 },
  ];
  return (
    <div className="mt-6 space-y-2.5">
      {rows.map((r, i) => (
        <div key={r.label} className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-24">{r.label}</span>
          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-primary"
              initial={{ width: 0 }}
              whileInView={{ width: `${r.val * 4}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: i * 0.1 }}
            />
          </div>
          <span className="text-xs font-semibold text-primary w-10 text-right">+{r.val}</span>
        </div>
      ))}
    </div>
  );
}

function PreferenceVector() {
  const tags = [
    { label: "Photography", weight: 0.9 },
    { label: "Food", weight: 0.7 },
    { label: "Hiking", weight: 0.55 },
    { label: "Nightlife", weight: 0.25 },
    { label: "Museums", weight: 0.6 },
  ];
  return (
    <div className="mt-6 space-y-2.5">
      {tags.map((t, i) => {
        const bars = 5;
        const filled = Math.round(t.weight * bars);
        return (
          <div key={t.label} className="flex items-center gap-3">
            <span className="text-xs w-24 font-medium">{t.label}</span>
            <div className="flex gap-1">
              {Array.from({ length: bars }).map((_, b) => (
                <motion.div
                  key={b}
                  initial={{ scaleY: 0 }}
                  whileInView={{ scaleY: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05 + b * 0.04 }}
                  className={`w-4 h-2.5 rounded-sm ${b < filled ? "bg-primary" : "bg-muted"}`}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function LiveProgress() {
  const items = [
    { label: "Searching flights", done: true },
    { label: "Checking hotels", done: true },
    { label: "Analyzing weather", done: false },
    { label: "Scoring options", done: false },
  ];
  return (
    <div className="mt-6 space-y-2.5">
      {items.map((it, i) => (
        <motion.div
          key={it.label}
          initial={{ opacity: 0, x: -8 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.12 }}
          className="flex items-center gap-3 bg-background rounded-xl px-3 py-2.5"
        >
          {it.done ? (
            <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center">
              <Check className="w-3 h-3 text-primary-foreground" />
            </div>
          ) : (
            <div className="w-5 h-5 rounded-full border-2 border-primary relative flex items-center justify-center">
              {i === 2 && <span className="absolute inset-0 rounded-full bg-primary/40 animate-ping" />}
            </div>
          )}
          <span className={`text-sm ${it.done ? "text-muted-foreground line-through" : "font-medium"}`}>{it.label}</span>
        </motion.div>
      ))}
    </div>
  );
}

// ---------------- Testimonials ----------------
const TESTIMONIALS = [
  { name: "Priya Sharma", style: "Digital nomad", avatar: "https://i.pravatar.cc/150?img=1", quote: "It remembers I hate red-eyes and love rainy cafes. Every plan now feels eerily personal." },
  { name: "Arjun Mehta", style: "Solo backpacker", avatar: "https://i.pravatar.cc/150?img=11", quote: "Planned a 10-day Himachal trip under ₹35k — it explained why Spiti beat Manali for October. Felt like a local guide." },
  { name: "Ananya Iyer", style: "Family of 4", avatar: "https://i.pravatar.cc/150?img=20", quote: "Balanced kids, budget, and weather for our Kerala trip. The ranked plan saved us a week of WhatsApp group chaos." },
  { name: "Rohan Gupta", style: "Weekend hopper", avatar: "https://i.pravatar.cc/150?img=13", quote: "Compared 14 itineraries in seconds and showed its math. Booked a Goa long weekend with full confidence." },
  { name: "Sneha Reddy", style: "Honeymoon planner", avatar: "https://i.pravatar.cc/150?img=31", quote: "Three ranked plans, each with a clear 'why.' We picked Rajasthan over Bali and had the trip of our lives." },
  { name: "Vikram Nair", style: "Business + leisure", avatar: "https://i.pravatar.cc/150?img=33", quote: "Stitched two client cities and a weekend in Udaipur into one route. It reasoned the layovers better than my travel desk." },
  { name: "Meera Krishnan", style: "Slow traveler", avatar: "https://i.pravatar.cc/150?img=45", quote: "I wanted less-crowded hill towns in monsoon. It ranked Coorg over Munnar with a clear weather-vs-crowd trade-off." },
  { name: "Aditya Joshi", style: "Group of friends", avatar: "https://i.pravatar.cc/150?img=14", quote: "Six of us, one budget, zero fights. The ‘why this won’ breakdown settled every argument instantly." },
  { name: "Marcus Rivera", style: "Adventure seeker", avatar: "https://i.pravatar.cc/150?img=8", quote: "The live streaming of agents working is mesmerizing — hotels, weather, routes, all reasoned in front of you." },
  { name: "Ana López", style: "Honeymoon planner", avatar: "https://i.pravatar.cc/150?img=9", quote: "Three ranked plans, each with a clear rationale. We chose with full confidence and never looked back." },
];

function TestimonialCard({ t }: { t: typeof TESTIMONIALS[number] }) {
  return (
    <div className="shrink-0 w-[320px] bg-card rounded-3xl p-6 shadow-sm">
      <div className="flex items-center gap-3 mb-4">
        <img src={t.avatar} alt={t.name} className="w-12 h-12 rounded-full object-cover" />
        <div>
          <div className="font-semibold text-sm">{t.name}</div>
          <div className="text-xs text-muted-foreground">{t.style}</div>
        </div>
      </div>
      <div className="flex gap-0.5 mb-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Star key={i} className="w-3.5 h-3.5 fill-primary text-primary" />
        ))}
      </div>
      <p className="text-sm leading-relaxed text-foreground/80">"{t.quote}"</p>
    </div>
  );
}

// ---------------- Page ----------------
const Landing = () => {
  const [wordIndex, setWordIndex] = useState(0);
  const [mouse, setMouse] = useState({ x: 0, y: 0 });
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const interval = setInterval(() => setWordIndex((p) => (p + 1) % ROTATING.length), 2500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const el = heroRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const nx = (e.clientX - cx) / (rect.width / 2);
      const ny = (e.clientY - cy) / (rect.height / 2);
      setMouse({ x: Math.max(-1, Math.min(1, nx)), y: Math.max(-1, Math.min(1, ny)) });
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <style>{`
        @keyframes nomad-float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-12px); }
        }
      `}</style>

      {/* Hero */}
      <section ref={heroRef} className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 py-20 lg:py-28">
          <div className="relative min-h-[620px] flex items-center justify-center">
            {/* Floating geometric shapes */}
            <div className="absolute inset-0 pointer-events-none" aria-hidden>
              {FLOAT_SHAPES.map((s, i) => <FloatingShape key={i} s={s} />)}
            </div>

            {/* Destination cards */}
            {DESTINATIONS.map((d, i) => (
              <DestinationCard key={d.name} dest={d} idx={i} mouse={mouse} />
            ))}

            {/* Centerpiece */}
            <div className="relative z-10 text-center max-w-3xl mx-auto px-2">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold mb-6"
              >
                <Sparkles className="w-3.5 h-3.5" />
                AI travel consultant
              </motion.div>

              <h1 className="font-display font-bold tracking-[-0.02em] text-5xl md:text-6xl lg:text-7xl leading-[1.05] text-foreground">
                Plan smarter.<br />
                Travel{" "}
                <span className="inline-block relative overflow-hidden" style={{ height: "1.05em", minWidth: "5ch", verticalAlign: "bottom" }}>
                  <AnimatePresence mode="wait">
                    <motion.span
                      key={wordIndex}
                      initial={{ y: "100%", opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      exit={{ y: "-100%", opacity: 0 }}
                      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                      className="inline-block text-primary leading-[1.05]"
                    >
                      {ROTATING[wordIndex]}
                    </motion.span>
                  </AnimatePresence>
                </span>
              </h1>

              <p className="mt-6 text-base md:text-lg text-muted-foreground max-w-xl mx-auto leading-relaxed">
                Wayfinder reasons about trade-offs and explains every choice — like a consultant, not a chatbot.
              </p>

              <div className="mt-8 flex justify-center">
                <Button size="lg" className="text-base font-semibold h-12 px-7" asChild>
                  <Link to="/plan">
                    Plan my trip <ArrowRight className="w-4 h-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="relative py-20 lg:py-28">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-14">
            <h2 className="font-display font-bold text-4xl md:text-5xl tracking-[-0.02em]">How it works</h2>
            <p className="mt-3 text-muted-foreground">Three steps from idea to ranked itinerary.</p>
          </div>

          <div className="relative">
            {/* Dashed connector (desktop) */}
            <div className="hidden lg:block absolute top-1/2 left-[8%] right-[8%] h-px -translate-y-1/2 border-t-2 border-dashed border-border" aria-hidden />

            <div className="relative flex flex-col lg:flex-row gap-6 lg:gap-8">
              <StepCard index={0} tint="bg-[hsl(340,90%,96%)]">
                <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-4">
                  <MessageSquare className="w-5 h-5" />
                </div>
                <div className="text-xs font-semibold text-primary mb-1">Step 1</div>
                <h3 className="font-display font-bold text-xl mb-2">Tell us your trip</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Just type it. "10 days in Japan, photography focus, mid-budget, hate crowds."
                </p>
              </StepCard>

              <StepCard index={1} tint="bg-[hsl(160,55%,93%)]">
                <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-4">
                  <Workflow className="w-5 h-5" />
                </div>
                <div className="text-xs font-semibold text-primary mb-1">Step 2</div>
                <h3 className="font-display font-bold text-xl mb-2">Agents analyze everything</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Specialized agents reason in parallel across every dimension.
                </p>
                <AgentPipeline />
              </StepCard>

              <StepCard index={2} tint="bg-[hsl(40,95%,93%)]">
                <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-4">
                  <ListChecks className="w-5 h-5" />
                </div>
                <div className="text-xs font-semibold text-primary mb-1">Step 3</div>
                <h3 className="font-display font-bold text-xl mb-2">Get your ranked plan</h3>
                <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                  Three options, scored and explained — pick with full confidence.
                </p>
                <div className="bg-background rounded-xl p-3 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold">A · Kyoto loop</span>
                    <span className="text-primary font-bold">92</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold">B · Osaka mix</span>
                    <span className="text-primary font-bold">88</span>
                  </div>
                </div>
              </StepCard>
            </div>
          </div>
        </div>
      </section>

      {/* Bento features */}
      <section className="py-20 lg:py-28 bg-muted/40">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-14">
            <h2 className="font-display font-bold text-4xl md:text-5xl tracking-[-0.02em]">Built like a consultant</h2>
            <p className="mt-3 text-muted-foreground">Ranked answers, transparent reasoning, real memory.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 auto-rows-[minmax(260px,auto)]">
            <BentoTile index={0} tint="bg-[hsl(340,90%,96%)]" className="md:row-span-2">
              <div className="text-xs font-semibold text-primary mb-2">Ranked options</div>
              <h3 className="font-display font-bold text-2xl md:text-3xl tracking-[-0.01em]">Ranked options, not one answer</h3>
              <p className="text-sm text-muted-foreground mt-2 max-w-md">
                Every plan gets three candidates, each scored on what matters to you.
              </p>
              <RankedScores />
            </BentoTile>

            <BentoTile index={1} tint="bg-[hsl(160,55%,93%)]">
              <div className="text-xs font-semibold text-primary mb-2">Explainability</div>
              <h3 className="font-display font-bold text-xl md:text-2xl tracking-[-0.01em]">Why it chose Kyoto</h3>
              <Explainability />
            </BentoTile>

            <BentoTile index={2} tint="bg-[hsl(255,70%,95%)]">
              <div className="text-xs font-semibold text-primary mb-2">Memory</div>
              <h3 className="font-display font-bold text-xl md:text-2xl tracking-[-0.01em]">Remembers you</h3>
              <PreferenceVector />
            </BentoTile>

            <BentoTile index={3} tint="bg-[hsl(40,95%,93%)]" className="md:row-span-2 md:col-start-2 md:row-start-2">
              <div className="text-xs font-semibold text-primary mb-2">Live</div>
              <h3 className="font-display font-bold text-2xl md:text-3xl tracking-[-0.01em]">Live progress streaming</h3>
              <p className="text-sm text-muted-foreground mt-2 max-w-md">
                Watch each specialist agent work in real time — no black box.
              </p>
              <LiveProgress />
            </BentoTile>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-20 lg:py-28 overflow-hidden">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 mb-12">
          <div className="text-center max-w-2xl mx-auto">
            <h2 className="font-display font-bold text-4xl md:text-5xl tracking-[-0.02em]">Loved by travelers</h2>
            <p className="mt-3 text-muted-foreground">From solo backpackers to families of four.</p>
          </div>
        </div>

        <div className="group relative">
          <div className="flex gap-5 w-max animate-[nomad-marquee_40s_linear_infinite] group-hover:[animation-play-state:paused]">
            {[...TESTIMONIALS, ...TESTIMONIALS].map((t, i) => (
              <TestimonialCard key={i} t={t} />
            ))}
          </div>
        </div>
        <style>{`
          @keyframes nomad-marquee {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
          }
        `}</style>
      </section>

      {/* CTA footer */}
      <section className="py-12 lg:py-20 px-6 lg:px-8">
        <div className="max-w-6xl mx-auto relative bg-foreground text-background rounded-[2.5rem] px-6 py-16 lg:px-16 lg:py-24 overflow-hidden text-center">
          {/* Floating shapes inside */}
          <div className="absolute inset-0 pointer-events-none opacity-80" aria-hidden>
            {FLOAT_SHAPES.slice(0, 8).map((s, i) => <FloatingShape key={i} s={s} />)}
          </div>

          <div className="relative">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary mb-6">
              <Sparkles className="w-7 h-7 text-primary-foreground" />
            </div>
            <h2 className="font-display font-bold text-4xl md:text-5xl lg:text-6xl tracking-[-0.02em]">
              Ready to plan your<br />next adventure?
            </h2>
            <p className="mt-5 text-base md:text-lg text-background/70 max-w-xl mx-auto">
              No credit card. No fluff. Just smarter travel decisions.
            </p>
            <div className="mt-8 flex justify-center">
              <Button size="lg" className="text-base font-semibold h-12 px-7 bg-primary text-primary-foreground hover:bg-primary/90" asChild>
                <Link to="/plan">
                  Start planning free <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      <footer className="py-10 text-center text-sm text-muted-foreground">
        © {new Date().getFullYear()} Wayfinder · Plan smarter, travel better.
      </footer>
    </div>
  );
};

export default Landing;
