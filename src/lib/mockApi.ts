// Single source of truth for the Wayfinder prototype's data layer.
// The async functions at the bottom call the FastAPI backend with the user's
// Supabase JWT. The exported function/type signatures are unchanged from the
// mock prototype, so the component layer consumes them without modification.
// The MOCK_* fixtures below are retained as static UI seed data still imported
// directly by some screens.

import {
  Camera,
  Users,
  CloudSun,
  Wallet,
  PartyPopper,
  MapPin,
  Map as MapIcon,
} from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

// ----------------------------- Types -----------------------------

export type ProgressStep = { label: string; status: "done" | "active" | "pending" };

export type MicroStat = { icon: React.ComponentType<{ className?: string }>; label: string };
export type BreakdownRow = { label: string; val: number; color: string };
export type FlightOpt = { airline: string; price: string; duration: string; tag: string };
export type HotelOpt = { name: string; rating: number; price: string; distance: string; tag: string };
export type EventOpt = { name: string; date: string; type: string };
export type DayPlan = { day: string; title: string; items: string[] };

// Decision-trace ("Why X Won") breakdown returned by the backend results payload.
// Each factor is one feature's additive contribution; the contributions sum to
// the trace's final score (within the engine's 0.001 tolerance).
export type FeatureContribution = { feature: string; contribution: number };
export type DecisionTrace = {
  destination: string;
  score: number;
  factors: FeatureContribution[];
};

export type Result = {
  grade: "A" | "B" | "C";
  name: string;
  country: string;
  flag: string;
  score: number;
  image: string;
  micro: MicroStat[];
  teaser: string;
  breakdown: BreakdownRow[];
  decisionTrace?: DecisionTrace;
  itinerary: DayPlan[];
  flights: FlightOpt[];
  hotels: HotelOpt[];
  events: EventOpt[];
  badge?: string;
};

export type Preference = { topic: string; weight: number };

export type Trip = {
  id: string;
  destination: string;
  country: string;
  image: string;
  startDate: string;
  endDate: string;
  durationDays: number;
  budgetUsed: string;
  budgetTotal: string;
  topSpot: string;
};

// ----------------------------- Fixtures -----------------------------

export const BAR_COLORS = [
  "hsl(340 75% 58%)",
  "hsl(45 90% 55%)",
  "hsl(170 60% 45%)",
  "hsl(250 60% 60%)",
  "hsl(20 85% 60%)",
  "hsl(200 70% 50%)",
];

export const MOCK_RESULTS: Result[] = [
  {
    grade: "A",
    name: "Kyoto + Hakone loop",
    country: "Japan",
    flag: "🇯🇵",
    score: 92,
    image: "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=1200&q=80",
    micro: [
      { icon: Camera, label: "Photography" },
      { icon: Users, label: "Low crowd" },
      { icon: CloudSun, label: "Mild weather" },
    ],
    teaser: "Quiet temples at dawn, photogenic alleys, soaking onsen at dusk.",
    breakdown: [
      { label: "Photography", val: 18, color: BAR_COLORS[0] },
      { label: "Crowd Density", val: 15, color: BAR_COLORS[1] },
      { label: "Weather", val: 14, color: BAR_COLORS[2] },
      { label: "Food", val: 12, color: BAR_COLORS[3] },
      { label: "Travel Eff.", val: 10, color: BAR_COLORS[4] },
      { label: "Budget Fit", val: 8, color: BAR_COLORS[5] },
    ],
    badge: "Best for you",
    decisionTrace: {
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
    },
    itinerary: [
      { day: "Day 1", title: "Arrive Kyoto · Gion stroll", items: ["Check in to ryokan in Higashiyama", "Evening walk through Gion's lantern alleys", "Kaiseki dinner"] },
      { day: "Day 2", title: "Arashiyama bamboo + Fushimi", items: ["Dawn shoot at Arashiyama bamboo grove", "Tenryu-ji temple gardens", "Sunset at Fushimi Inari (avoid midday crowds)"] },
      { day: "Day 3", title: "Nara day trip", items: ["Train to Nara (45 min)", "Todai-ji temple & deer park", "Return for sushi omakase in Pontocho"] },
      { day: "Day 4", title: "Kyoto → Hakone", items: ["Shinkansen to Odawara", "Hakone open-air sculpture museum", "Onsen ryokan check-in with Fuji view"] },
      { day: "Day 5", title: "Hakone loop · Return", items: ["Lake Ashi pirate ship", "Owakudani volcanic valley", "Return to Tokyo for departure"] },
    ],
    flights: [
      { airline: "ANA NH 829", price: "₹68,400", duration: "11h 25m", tag: "Best balance" },
      { airline: "JAL JL 740", price: "₹62,200", duration: "12h 50m", tag: "Cheapest" },
      { airline: "SQ 11 via SIN", price: "₹74,900", duration: "10h 10m", tag: "Shortest" },
    ],
    hotels: [
      { name: "Hoshinoya Kyoto", rating: 5, price: "₹38,000/n", distance: "12 min to center", tag: "Most photogenic" },
      { name: "Nazuna Kyoto Gosho", rating: 4, price: "₹14,500/n", distance: "8 min to center", tag: "Best value" },
      { name: "The Thousand Kyoto", rating: 5, price: "₹22,000/n", distance: "0 min · at station", tag: "Most convenient" },
    ],
    events: [
      { name: "Jidai Matsuri parade", date: "Oct 22", type: "Festival" },
      { name: "Kurama Fire Festival", date: "Oct 22", type: "Festival" },
      { name: "Kyoto Photo Biennale", date: "Oct 18 – Nov 5", type: "Exhibition" },
      { name: "Tofuku-ji night illumination", date: "Oct 25", type: "Exhibition" },
    ],
  },
  {
    grade: "B",
    name: "Osaka + Nara mix",
    country: "Japan",
    flag: "🇯🇵",
    score: 88,
    image: "https://images.unsplash.com/photo-1590559899731-a382839e5549?w=1200&q=80",
    micro: [
      { icon: Wallet, label: "Best budget" },
      { icon: PartyPopper, label: "Street food" },
      { icon: MapPin, label: "Easy transit" },
    ],
    teaser: "Street food capital with easy day trips. Unbeatable on cost-per-day.",
    breakdown: [
      { label: "Food", val: 19, color: BAR_COLORS[3] },
      { label: "Budget Fit", val: 16, color: BAR_COLORS[5] },
      { label: "Travel Eff.", val: 14, color: BAR_COLORS[4] },
      { label: "Photography", val: 11, color: BAR_COLORS[0] },
      { label: "Weather", val: 12, color: BAR_COLORS[2] },
      { label: "Crowd Density", val: 6, color: BAR_COLORS[1] },
    ],
    decisionTrace: {
      destination: "Osaka + Nara mix",
      score: 88,
      factors: [
        { feature: "Food", contribution: 22 },
        { feature: "Budget Fit", contribution: 18 },
        { feature: "Travel Efficiency", contribution: 16 },
        { feature: "Weather", contribution: 13 },
        { feature: "Photography", contribution: 12 },
        { feature: "Crowd Density", contribution: 7 },
      ],
    },
    itinerary: [
      { day: "Day 1", title: "Arrive Osaka · Dotonbori", items: ["Check in to Namba hotel", "Dotonbori takoyaki crawl", "Glico sign at night"] },
      { day: "Day 2", title: "Osaka castle + Umeda", items: ["Morning at Osaka castle", "Umeda Sky Building views", "Kushikatsu in Shinsekai"] },
      { day: "Day 3", title: "Nara day trip", items: ["Train to Nara", "Todai-ji & deer park", "Mochi pounding at Nakatanidou"] },
      { day: "Day 4", title: "Kobe day trip", items: ["Kobe beef lunch", "Kitano-cho district", "Harborland sunset"] },
      { day: "Day 5", title: "Return", items: ["Last-minute Kuromon market", "Depart from KIX"] },
    ],
    flights: [
      { airline: "JAL JL 727", price: "₹61,800", duration: "13h 05m", tag: "Cheapest" },
      { airline: "Cathay CX 567", price: "₹66,500", duration: "11h 40m", tag: "Best balance" },
    ],
    hotels: [
      { name: "Cross Hotel Osaka", rating: 4, price: "₹9,800/n", distance: "5 min to Dotonbori", tag: "Best value" },
      { name: "Conrad Osaka", rating: 5, price: "₹26,400/n", distance: "Skyline views", tag: "Splurge" },
    ],
    events: [
      { name: "Midosuji Autumn Party", date: "Oct 26", type: "Festival" },
      { name: "Osaka Asian Film Fest", date: "Oct 20 – 28", type: "Exhibition" },
    ],
  },
  {
    grade: "C",
    name: "Hokkaido nature",
    country: "Japan",
    flag: "🇯🇵",
    score: 84,
    image: "https://images.unsplash.com/photo-1542640244-7e672d6cef4e?w=1200&q=80",
    micro: [
      { icon: Users, label: "Least crowded" },
      { icon: MapIcon, label: "Off-beat" },
      { icon: Camera, label: "Autumn colors" },
    ],
    teaser: "Crisp autumn forests, remote ryokan, most off-the-beaten-path.",
    breakdown: [
      { label: "Crowd Density", val: 20, color: BAR_COLORS[1] },
      { label: "Photography", val: 17, color: BAR_COLORS[0] },
      { label: "Weather", val: 9, color: BAR_COLORS[2] },
      { label: "Food", val: 13, color: BAR_COLORS[3] },
      { label: "Travel Eff.", val: 5, color: BAR_COLORS[4] },
      { label: "Budget Fit", val: 10, color: BAR_COLORS[5] },
    ],
    decisionTrace: {
      destination: "Hokkaido nature",
      score: 84,
      factors: [
        { feature: "Crowd Density", contribution: 23 },
        { feature: "Photography", contribution: 19 },
        { feature: "Food", contribution: 15 },
        { feature: "Budget Fit", contribution: 11 },
        { feature: "Weather", contribution: 10 },
        { feature: "Travel Efficiency", contribution: 6 },
      ],
    },
    itinerary: [
      { day: "Day 1", title: "Arrive Sapporo", items: ["Susukino izakaya night", "Miso ramen at Sumire"] },
      { day: "Day 2", title: "Otaru canal town", items: ["Glass workshops on Sakaimachi", "Fresh uni at Sankaku market"] },
      { day: "Day 3", title: "Daisetsuzan autumn drive", items: ["Asahidake ropeway", "Sounkyo gorge red leaves"] },
      { day: "Day 4", title: "Furano + Biei", items: ["Patchwork hills photography", "Blue Pond at dusk"] },
      { day: "Day 5", title: "Return via Sapporo", items: ["Last bowl of ramen", "Depart from CTS"] },
    ],
    flights: [
      { airline: "ANA NH 60", price: "₹71,200", duration: "13h 50m", tag: "Best balance" },
      { airline: "Air Japan via NRT", price: "₹64,000", duration: "15h 20m", tag: "Cheapest" },
    ],
    hotels: [
      { name: "Sounkyo Onsen Ryokan", rating: 4, price: "₹16,800/n", distance: "Gorge-front", tag: "Most scenic" },
      { name: "JR Tower Hotel Sapporo", rating: 4, price: "₹12,400/n", distance: "At station", tag: "Most convenient" },
    ],
    events: [
      { name: "Sounkyo Momiji Festival", date: "Oct 19 – 21", type: "Festival" },
      { name: "Sapporo Autumn Fest", date: "Oct 1 – 31", type: "Festival" },
      { name: "PMF chamber concert", date: "Oct 23", type: "Concert" },
    ],
  },
];

export const MOCK_TRIPS: Trip[] = [
  {
    id: "kyoto-oct-2024",
    destination: "Japan loop",
    country: "Japan",
    image: "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=1200&q=80",
    startDate: "Oct 18",
    endDate: "Oct 28, 2024",
    durationDays: 10,
    budgetUsed: "₹1.62L",
    budgetTotal: "₹2L",
    topSpot: "Kyoto",
  },
  {
    id: "lisbon-mar-2024",
    destination: "Portugal coast",
    country: "Portugal",
    image: "https://images.unsplash.com/photo-1513735492246-483525079686?w=1200&q=80",
    startDate: "Mar 4",
    endDate: "Mar 12, 2024",
    durationDays: 8,
    budgetUsed: "₹1.45L",
    budgetTotal: "₹1.5L",
    topSpot: "Lisbon",
  },
  {
    id: "iceland-aug-2023",
    destination: "Ring road",
    country: "Iceland",
    image: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1200&q=80",
    startDate: "Aug 12",
    endDate: "Aug 22, 2023",
    durationDays: 10,
    budgetUsed: "₹3.10L",
    budgetTotal: "₹3L",
    topSpot: "Reykjavík",
  },
  {
    id: "rajasthan-dec-2022",
    destination: "Heritage circuit",
    country: "India",
    image: "https://images.unsplash.com/photo-1599661046289-e31897846e41?w=1200&q=80",
    startDate: "Dec 22",
    endDate: "Dec 30, 2022",
    durationDays: 8,
    budgetUsed: "₹38K",
    budgetTotal: "₹40K",
    topSpot: "Udaipur",
  },
];

export const MOCK_PREFERENCES: Preference[] = [
  { topic: "Photography", weight: 0.82 },
  { topic: "Local Food", weight: 0.74 },
  { topic: "Anime", weight: 0.61 },
  { topic: "Nature", weight: 0.55 },
  { topic: "Luxury", weight: 0.22 },
  { topic: "Adventure", weight: 0.48 },
  { topic: "History", weight: 0.66 },
  { topic: "Nightlife", weight: 0.30 },
];

// ----------------------------- Real API client -----------------------------

// Base URL of the FastAPI backend. Defaults to same-origin so the dev proxy or a
// reverse proxy can forward `/api/*`; override with VITE_API_BASE_URL in env.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

/** Resolve the current Supabase access token (JWT) for the signed-in user. */
async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

/**
 * Authenticated fetch against the backend. Attaches `Authorization: Bearer <JWT>`
 * from the active Supabase session and parses the JSON response. Throws on
 * non-2xx responses (callers may catch to render error states).
 */
async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = await getAccessToken();
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    throw new Error(`Request to ${path} failed with status ${res.status}`);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

// ----------------------------- API functions -----------------------------

// POST /api/trips/plan
export async function planTrip(body: { prompt: string; interests?: string[] }) {
  return apiFetch<{ tripId: string; status: string }>("/api/trips/plan", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// GET /api/trips/:id/results
//
// The backend returns a single results object (TripResultsDTO): a ranked
// `destinations[]` plus the shared itinerary/flights/hotels/events for the
// recommended plan, the chosen destination's decision trace, and a
// human-readable `narration` (deterministic ranking facts plus live LLM prose).
// The prototype's component layer consumes `Result[]`, so this adapter maps the
// backend shape onto the existing `Result` cards and surfaces the narration
// alongside them.

/** Backend `GET /results` payload shape (aligned to app.models.schemas DTOs). */
type ApiFactor = { feature: string; contribution: number };
type ApiDestination = {
  name: string;
  country: string;
  flag: string;
  score: number;
  rank: string; // "A" | "B" | "C" | "#n"
  factors: ApiFactor[];
};
type ApiFlight = { airline: string; price: number; duration: string; rationale: string };
type ApiHotel = {
  name: string;
  rating: number;
  pricePerNight: number;
  distanceKm: number;
  rationale: string;
};
type ApiActivity = { name: string; time: string; cost: number };
type ApiDay = { day: number; date: string; activities: ApiActivity[] };
type ApiResults = {
  destinations: ApiDestination[];
  itinerary: ApiDay[];
  flights: ApiFlight[];
  hotels: ApiHotel[];
  events: EventOpt[];
  decisionTrace?: DecisionTrace | null;
  narration?: string;
};

/** Ranked results plus the whole-ranking narration for the results screen. */
export type ResultsBundle = { results: Result[]; narration: string };

const RESULT_IMAGES = [
  "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=1200&q=80",
  "https://images.unsplash.com/photo-1590559899731-a382839e5549?w=1200&q=80",
  "https://images.unsplash.com/photo-1542640244-7e672d6cef4e?w=1200&q=80",
];

/** Format a numeric rupee amount as a compact "₹68,400" string. */
function formatINR(amount: number): string {
  return `₹${Math.round(amount).toLocaleString("en-IN")}`;
}

/** Turn an engine feature key ("budget_fit") into a label ("Budget Fit"). */
function prettyFeature(feature: string): string {
  return feature
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Adapt the backend results payload onto the prototype's `Result[]` + narration. */
function adaptApiResults(dto: ApiResults): ResultsBundle {
  // The itinerary/flights/hotels/events describe the recommended plan and are
  // shared across the ranked options by the backend; attach them to each card
  // so the detail tabs are populated for whichever option is selected.
  const flights: FlightOpt[] = (dto.flights ?? []).map((f) => ({
    airline: f.airline,
    price: formatINR(f.price),
    duration: f.duration,
    tag: f.rationale,
  }));
  const hotels: HotelOpt[] = (dto.hotels ?? []).map((h) => ({
    name: h.name,
    rating: h.rating,
    price: `${formatINR(h.pricePerNight)}/n`,
    distance: `${h.distanceKm} km`,
    tag: h.rationale,
  }));
  const events: EventOpt[] = dto.events ?? [];
  const itinerary: DayPlan[] = (dto.itinerary ?? []).map((d) => ({
    day: `Day ${d.day}`,
    title: d.date || "",
    items: d.activities.map((a) => a.name),
  }));

  const grades: Array<"A" | "B" | "C"> = ["A", "B", "C"];
  const results: Result[] = (dto.destinations ?? []).map((d, i) => {
    const grade = (["A", "B", "C"].includes(d.rank) ? d.rank : grades[i] ?? "C") as
      | "A"
      | "B"
      | "C";
    return {
      grade,
      name: d.name,
      country: d.country,
      flag: d.flag,
      score: Math.round(d.score),
      image: RESULT_IMAGES[i % RESULT_IMAGES.length],
      micro: [],
      teaser: "",
      breakdown: d.factors.map((f, idx) => ({
        label: prettyFeature(f.feature),
        val: Math.round(f.contribution * 10) / 10,
        color: BAR_COLORS[idx % BAR_COLORS.length],
      })),
      // Each option's own additive factors form its "Why X won" trace.
      decisionTrace: {
        destination: d.name,
        score: d.score,
        factors: d.factors,
      },
      itinerary,
      flights,
      hotels,
      events,
      badge: i === 0 ? "Best for you" : undefined,
    };
  });

  return { results, narration: dto.narration ?? "" };
}

export async function getResults(tripId: string): Promise<ResultsBundle> {
  const dto = await apiFetch<ApiResults>(
    `/api/trips/${encodeURIComponent(tripId)}/results`,
  );
  return adaptApiResults(dto);
}

// GET /api/trips
export async function getTrips(): Promise<Trip[]> {
  return apiFetch<Trip[]>("/api/trips");
}

// GET /api/trips/:id
export async function getTrip(id: string): Promise<Trip | undefined> {
  const token = await getAccessToken();
  const headers = new Headers({ Accept: "application/json" });
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const res = await fetch(`${API_BASE_URL}/api/trips/${encodeURIComponent(id)}`, { headers });
  if (res.status === 404) {
    return undefined;
  }
  if (!res.ok) {
    throw new Error(`Request to /api/trips/${id} failed with status ${res.status}`);
  }
  return (await res.json()) as Trip;
}

// GET /api/preferences
export async function getPreferences() {
  return apiFetch<{ preferences: Preference[] }>("/api/preferences");
}

// PATCH /api/preferences
export async function patchPreferences(body: { preferences: Preference[] }) {
  return apiFetch<{ preferences: Preference[] }>("/api/preferences", {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

// ----------------------------- Progress stream (WebSocket) -----------------------------

// Phase of a stage notification: a stage either starts or completes.
export type ProgressPhase = "start" | "complete";

/**
 * A single progress notification relayed over the WebSocket progress stream.
 * Mirrors the backend `ProgressEvent` (app/orchestration/graph.py): it carries
 * only stage metadata and a human-readable message — never a live/real-time
 * price (Requirement 13.2/13.3).
 */
export type ProgressEvent = { stage: string; phase: ProgressPhase; message: string };

/**
 * Ordered backend pipeline stage keys, mirroring `STAGES` in
 * `app/orchestration/graph.py`. Used to translate the count of completed stage
 * events into an overall completion fraction for the progress UI.
 */
export const PROGRESS_STAGES = [
  "coordinator",
  "destination",
  "decision_engine",
  "tools",
  "solver",
  "itinerary",
  "merge",
] as const;

export type ProgressStreamHandlers = {
  /** Fired for every stage event (start or complete). */
  onEvent?: (event: ProgressEvent) => void;
  /** Fired once when the run finishes and the socket closes cleanly. */
  onComplete?: () => void;
  /** Fired when the handshake/stream fails (e.g. auth rejected, code 1008). */
  onError?: (reason: { code: number }) => void;
};

/**
 * Subscribe to `WS /api/trips/{tripId}/progress` and relay each stage event to
 * the supplied handlers. The Supabase JWT is passed as a `token` query param
 * because browser WebSocket clients cannot set the `Authorization` header; the
 * backend handshake accepts the token from that query param.
 *
 * Returns an unsubscribe function that closes the socket and suppresses any
 * further callbacks. Safe to call before the socket has opened.
 */
export function subscribeToProgress(
  tripId: string,
  handlers: ProgressStreamHandlers,
): () => void {
  let socket: WebSocket | null = null;
  let closed = false;

  void (async () => {
    const token = await getAccessToken();
    if (closed) return;

    // Resolve the ws(s):// origin from the REST base (or same-origin) and flip
    // the http(s) scheme to ws(s).
    const httpBase = API_BASE_URL || window.location.origin;
    const wsBase = httpBase.replace(/^http/i, "ws");
    const url = new URL(`${wsBase}/api/trips/${encodeURIComponent(tripId)}/progress`);
    if (token) {
      url.searchParams.set("token", token);
    }

    socket = new WebSocket(url.toString());

    socket.onmessage = (e: MessageEvent) => {
      try {
        handlers.onEvent?.(JSON.parse(e.data) as ProgressEvent);
      } catch {
        // Ignore malformed frames rather than tearing down the stream.
      }
    };

    socket.onclose = (e: CloseEvent) => {
      if (closed) return;
      closed = true;
      // 1000 (normal) / 1005 (no status) signal a clean run completion; any
      // other code (e.g. 1008 policy violation from a rejected handshake) is an
      // error the caller can fall back from.
      if (e.code === 1000 || e.code === 1005) {
        handlers.onComplete?.();
      } else {
        handlers.onError?.({ code: e.code });
      }
    };
  })();

  return () => {
    closed = true;
    socket?.close();
  };
}
