import { Link, useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight, Calendar, MapPin, Sparkles, Plane } from "lucide-react";
import { MOCK_TRIPS, type Trip } from "@/lib/mockApi";

function TripCard({ trip, i }: { trip: Trip; i: number }) {
  const navigate = useNavigate();
  return (
    <motion.button
      type="button"
      onClick={() => navigate(`/trips/${trip.id}`)}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: i * 0.08, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ scale: 1.02 }}
      className="group text-left bg-card rounded-3xl overflow-hidden shadow-sm hover:shadow-xl transition-shadow"
    >
      <div className="relative aspect-[16/10] overflow-hidden">
        <img
          src={trip.image}
          alt={trip.destination}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />
        <span className="absolute top-3 left-3 inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-background/90 backdrop-blur text-[11px] font-semibold">
          <MapPin className="w-3 h-3 text-primary" /> {trip.topSpot}
        </span>
        <span className="absolute top-3 right-3 inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-background/90 backdrop-blur text-[11px] font-semibold tabular-nums">
          {trip.budgetUsed} <span className="text-muted-foreground">/ {trip.budgetTotal}</span>
        </span>
      </div>

      <div className="p-5">
        <h3 className="font-display font-bold text-lg tracking-[-0.01em]">
          {trip.destination}
        </h3>
        <div className="text-xs text-muted-foreground mt-0.5">{trip.country}</div>

        <div className="mt-4 flex items-center justify-between">
          <div className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <Calendar className="w-3.5 h-3.5" />
            {trip.startDate} – {trip.endDate}
          </div>
          <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[11px] font-semibold">
            {trip.durationDays} days
          </span>
        </div>
      </div>
    </motion.button>
  );
}

function EmptyState() {
  return (
    <div className="bg-card rounded-3xl p-12 text-center max-w-md mx-auto">
      <div className="relative mx-auto w-24 h-24 mb-6">
        <div className="absolute inset-0 rounded-full bg-primary/10" />
        <Plane className="absolute inset-0 m-auto w-10 h-10 text-primary -rotate-12" />
      </div>
      <h3 className="font-display font-bold text-xl">No trips yet</h3>
      <p className="text-sm text-muted-foreground mt-2 mb-6">
        Your planned adventures will live here.
      </p>
      <Button asChild className="font-semibold">
        <Link to="/plan">
          Plan your first one <ArrowRight className="w-4 h-4" />
        </Link>
      </Button>
    </div>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <div className="h-16" aria-hidden />
      <main className="px-6 py-10 md:py-16">{children}</main>
    </div>
  );
}

export default function Trips() {
  const trips = MOCK_TRIPS;
  return (
    <PageShell>
      <div className="max-w-5xl mx-auto">
        <div className="flex items-end justify-between flex-wrap gap-4 mb-10">
          <div>
            <h1 className="font-display font-bold text-4xl md:text-5xl tracking-[-0.02em]">
              My trips
            </h1>
            <p className="mt-2 text-muted-foreground">
              Everywhere you've been with Wayfinder.
            </p>
          </div>
          <Button asChild className="font-semibold">
            <Link to="/plan">
              Plan a trip <ArrowRight className="w-4 h-4" />
            </Link>
          </Button>
        </div>

        {trips.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {trips.map((t, i) => (
              <TripCard key={t.id} trip={t} i={i} />
            ))}
          </div>
        )}
      </div>
    </PageShell>
  );
}

export function TripDetail() {
  const { id } = useParams();
  const trip = MOCK_TRIPS.find((t) => t.id === id);

  return (
    <PageShell>
      <div className="max-w-3xl mx-auto">
        <Button variant="ghost" asChild className="mb-6 font-medium -ml-2">
          <Link to="/trips">
            <ArrowLeft className="w-4 h-4" /> All trips
          </Link>
        </Button>

        {!trip ? (
          <div className="bg-card rounded-3xl p-10 text-center">
            <h2 className="font-display font-bold text-2xl">Trip not found</h2>
            <p className="text-sm text-muted-foreground mt-2">
              It may have been removed or never existed.
            </p>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="bg-card rounded-3xl overflow-hidden shadow-sm"
          >
            <img src={trip.image} alt={trip.destination} className="w-full h-64 object-cover" />
            <div className="p-7">
              <h1 className="font-display font-bold text-3xl tracking-[-0.02em]">
                {trip.destination}
              </h1>
              <div className="text-sm text-muted-foreground mt-1">{trip.country}</div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
                <Stat label="Dates" value={`${trip.startDate} – ${trip.endDate}`} />
                <Stat label="Duration" value={`${trip.durationDays} days`} />
                <Stat label="Budget" value={`${trip.budgetUsed} / ${trip.budgetTotal}`} />
                <Stat label="Top spot" value={trip.topSpot} />
              </div>

              <p className="mt-6 text-sm text-muted-foreground leading-relaxed">
                Full itinerary, flights, hotels, and reasoning for this trip will appear here.
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </PageShell>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-background rounded-2xl p-3">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold mt-0.5">{value}</div>
    </div>
  );
}
