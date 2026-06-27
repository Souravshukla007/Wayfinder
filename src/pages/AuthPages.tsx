import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { z } from "zod";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { ArrowRight, Loader2, Sparkles, Eye, EyeOff } from "lucide-react";

type Mode = "login" | "signup";

const QUOTES = [
  { quote: "Wayfinder reasoned through 14 itineraries in seconds. Felt like having a travel consultant on call.", author: "Sarah Chen", role: "Solo backpacker" },
  { quote: "Three ranked plans, each with a clear 'why.' We picked C with full confidence.", author: "Ana López", role: "Honeymoon planner" },
  { quote: "Finally a planner that respects trade-offs — budget vs. crowd vs. weather.", author: "Marcus Rivera", role: "Family of four" },
];

function RotatingQuote() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((p) => (p + 1) % QUOTES.length), 4000);
    return () => clearInterval(t);
  }, []);
  const q = QUOTES[i];
  return (
    <div className="relative h-32">
      <AnimatePresence mode="wait">
        <motion.figure
          key={i}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.5 }}
        >
          <blockquote className="font-display text-xl lg:text-2xl leading-snug tracking-[-0.01em]">
            "{q.quote}"
          </blockquote>
          <figcaption className="mt-4 text-sm opacity-75">
            <span className="font-semibold">{q.author}</span> · {q.role}
          </figcaption>
        </motion.figure>
      </AnimatePresence>
    </div>
  );
}

const FLOAT_SHAPES_AUTH = [
  { left: "12%", top: "24%", size: 10, type: "circle" as const, color: "hsl(var(--primary))", dur: 5 },
  { left: "82%", top: "18%", size: 12, type: "square" as const, color: "#FFD93D", dur: 6 },
  { left: "72%", top: "68%", size: 10, type: "triangle" as const, color: "#6BCB77", dur: 5.5 },
  { left: "18%", top: "76%", size: 9, type: "circle" as const, color: "#4D96FF", dur: 6.5 },
  { left: "46%", top: "12%", size: 8, type: "square" as const, color: "hsl(var(--primary))", dur: 5 },
  { left: "60%", top: "40%", size: 7, type: "circle" as const, color: "#FF9F43", dur: 7 },
];

function FloatShape({ s, i }: { s: typeof FLOAT_SHAPES_AUTH[number]; i: number }) {
  let shape: React.ReactNode;
  if (s.type === "circle") {
    shape = <div style={{ width: s.size, height: s.size, borderRadius: "50%", background: s.color }} />;
  } else if (s.type === "square") {
    shape = <div style={{ width: s.size, height: s.size, borderRadius: 2, background: s.color }} />;
  } else {
    shape = (
      <div
        style={{
          width: 0,
          height: 0,
          borderLeft: `${s.size / 2}px solid transparent`,
          borderRight: `${s.size / 2}px solid transparent`,
          borderBottom: `${s.size}px solid ${s.color}`,
        }}
      />
    );
  }
  return (
    <motion.div
      className="absolute"
      style={{ left: s.left, top: s.top }}
      animate={{ y: [0, -12, 0], rotate: s.type === "square" ? [0, 18, 0] : 0 }}
      transition={{ duration: s.dur, repeat: Infinity, ease: "easeInOut", delay: i * 0.4 }}
    >
      {shape}
    </motion.div>
  );
}

const LOGIN_BG = "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=1400&q=80"; // Kyoto
const SIGNUP_BG = "https://images.unsplash.com/photo-1514890547357-a9ee288728e0?w=1400&q=80"; // Venice

function IllustrationPanel({ mode }: { mode: Mode }) {
  const dests = ["Kyoto", "Venice", "Rio", "Giza", "Bali"];
  const bg = mode === "login" ? LOGIN_BG : SIGNUP_BG;
  return (
    <aside className="hidden lg:flex relative bg-foreground text-background flex-col justify-between p-10 overflow-hidden">
      {/* full-bleed travel photo + readability overlay */}
      <div aria-hidden className="absolute inset-0">
        <motion.img
          key={bg}
          src={bg}
          alt=""
          className="h-full w-full object-cover"
          initial={{ scale: 1.08, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
        />
        <div className="absolute inset-0 bg-foreground/80" />
        <div className="absolute inset-0 bg-gradient-to-t from-foreground via-foreground/70 to-foreground/40" />
      </div>

      {/* animated gradient orbs + grid + floating shapes */}
      <div aria-hidden className="absolute inset-0 pointer-events-none">
        <motion.div
          className="absolute top-[6%] left-[6%] w-56 h-56 rounded-full bg-primary/30 blur-3xl"
          animate={{ scale: [1, 1.25, 1], opacity: [0.45, 0.8, 0.45] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute bottom-[8%] right-[4%] w-72 h-72 rounded-full bg-primary/20 blur-3xl"
          animate={{ scale: [1.15, 1, 1.15], opacity: [0.35, 0.65, 0.35] }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        />
        <div
          className="absolute inset-0 opacity-[0.06]"
          style={{
            backgroundImage: "radial-gradient(hsl(var(--background)) 1px, transparent 1px)",
            backgroundSize: "22px 22px",
          }}
        />
        {FLOAT_SHAPES_AUTH.map((s, i) => (
          <FloatShape key={i} s={s} i={i} />
        ))}
      </div>

      <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="relative z-10">
        <Link to="/" className="relative inline-flex items-center gap-2 w-fit">
          <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-display font-bold text-primary text-2xl tracking-tight">Wayfinder</span>
        </Link>
      </motion.div>

      <div className="relative max-w-md">
        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          className="font-display font-bold text-3xl lg:text-[2.5rem] leading-[1.1] tracking-[-0.02em] mb-8"
        >
          Your next trip,
          <br />
          reasoned out loud.
        </motion.h2>

        <RotatingQuote />

        <div className="mt-10 flex flex-wrap gap-2">
          {dests.map((d, i) => (
            <motion.span
              key={d}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 + i * 0.1, type: "spring", stiffness: 300, damping: 20 }}
              className="inline-flex items-center rounded-full bg-background/10 backdrop-blur px-3 py-1 text-xs font-medium text-background/80 ring-1 ring-background/15"
            >
              {d}
            </motion.span>
          ))}
        </div>
      </div>

      <div className="relative text-xs opacity-60">Plan smarter. Travel deeper.</div>
    </aside>
  );
}

function FloatingInput({
  id,
  label,
  type = "text",
  value,
  onChange,
  error,
  autoComplete,
}: {
  id: string;
  label: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  error?: string;
  autoComplete?: string;
}) {
  const [focused, setFocused] = useState(false);
  const [visible, setVisible] = useState(false);
  const floated = focused || value.length > 0;
  // Password fields get a show/hide eye toggle; the rendered input type flips
  // between "password" and "text" while the declared field stays a password.
  const isPassword = type === "password";
  const inputType = isPassword ? (visible ? "text" : "password") : type;
  return (
    <div>
      <motion.div
        animate={{ scale: focused ? 1.015 : 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 24 }}
        className={`relative bg-card rounded-2xl transition-shadow ${
          error
            ? "ring-2 ring-destructive"
            : focused
              ? "ring-2 ring-primary/50 shadow-lg shadow-primary/10"
              : "ring-1 ring-border"
        }`}
      >
        <motion.label
          htmlFor={id}
          animate={{
            y: floated ? -8 : 8,
            scale: floated ? 0.85 : 1,
            color: error ? "hsl(var(--destructive))" : floated ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))",
          }}
          transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
          className="absolute left-4 top-3 origin-left text-sm font-medium pointer-events-none"
        >
          {label}
        </motion.label>
        <input
          id={id}
          type={inputType}
          value={value}
          autoComplete={autoComplete}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          className={`w-full h-14 pt-5 pb-1 px-4 bg-transparent rounded-2xl outline-none text-sm ${
            isPassword ? "pr-12" : ""
          }`}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setVisible((v) => !v)}
            aria-label={visible ? "Hide password" : "Show password"}
            aria-pressed={visible}
            tabIndex={-1}
            className="absolute right-3 top-1/2 -translate-y-1/2 inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
          >
            {visible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        )}
      </motion.div>
      <AnimatePresence initial={false}>
        {error && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <p className="text-xs text-destructive mt-1.5 px-1">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SubmitButton({ loading, label }: { loading: boolean; label: string }) {
  return (
    <motion.button
      type="submit"
      disabled={loading}
      animate={{ width: loading ? 56 : "100%" }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="h-12 bg-primary text-primary-foreground rounded-full font-semibold text-sm flex items-center justify-center gap-2 disabled:opacity-80 overflow-hidden mx-auto"
    >
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.span key="l" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Loader2 className="w-5 h-5 animate-spin" />
          </motion.span>
        ) : (
          <motion.span key="t" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="inline-flex items-center gap-2 whitespace-nowrap">
            {label} <ArrowRight className="w-4 h-4" />
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  );
}

function GoogleButton({ onError }: { onError: (m: string) => void }) {
  const [loading, setLoading] = useState(false);
  const handle = async () => {
    setLoading(true);
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: `${window.location.origin}/trips` },
      });
      if (error) {
        onError("Could not sign in with Google. Please try again.");
        setLoading(false);
        return;
      }
      // On success the browser redirects to Google; AuthContext picks up the
      // session on return and routes the user.
    } catch {
      onError("Could not sign in with Google. Please try again.");
      setLoading(false);
    }
  };
  return (
    <button
      type="button"
      onClick={handle}
      disabled={loading}
      className="w-full h-12 rounded-full bg-card ring-1 ring-border text-sm font-semibold inline-flex items-center justify-center gap-3 hover:bg-muted transition-colors disabled:opacity-60"
    >
      {loading ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.17-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.71v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.61z" />
          <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.81.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.71H.96v2.33A9 9 0 0 0 9 18z" />
          <path fill="#FBBC05" d="M3.97 10.71A5.4 5.4 0 0 1 3.68 9c0-.59.1-1.17.29-1.71V4.96H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.04l3.01-2.33z" />
          <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A9 9 0 0 0 .96 4.96l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z" />
        </svg>
      )}
      Continue with Google
    </button>
  );
}

const emailSchema = z.string().trim().email("Enter a valid email").max(255);
const passwordSchema = z.string().min(6, "At least 6 characters").max(128);
const nameSchema = z.string().trim().min(1, "Name is required").max(80);

function AuthForm({ mode }: { mode: Mode }) {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setFormError("");
    const next: Record<string, string> = {};

    const emailRes = emailSchema.safeParse(email);
    if (!emailRes.success) next.email = emailRes.error.issues[0].message;
    const pwRes = passwordSchema.safeParse(password);
    if (!pwRes.success) next.password = pwRes.error.issues[0].message;

    if (mode === "signup") {
      const nameRes = nameSchema.safeParse(name);
      if (!nameRes.success) next.name = nameRes.error.issues[0].message;
      if (confirm !== password) next.confirm = "Passwords don't match";
    }
    setErrors(next);
    if (Object.keys(next).length > 0) return;

    setLoading(true);
    try {
      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${window.location.origin}/trips`,
            data: { full_name: name },
          },
        });
        if (error) throw error;
        navigate("/trips");
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        navigate("/trips");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong";
      setFormError(msg);
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="w-full max-w-sm mx-auto space-y-4">
      <motion.div
        key={mode}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        className="text-center mb-2"
      >
        <motion.div
          initial={{ scale: 0, rotate: -30 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 260, damping: 18, delay: 0.05 }}
          className="mx-auto mb-4 inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10 text-primary"
        >
          <Sparkles className="h-5 w-5" />
        </motion.div>
        <h1 className="font-display font-bold text-3xl md:text-4xl tracking-[-0.02em]">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          {mode === "login"
            ? "Sign in to plan your next adventure."
            : "Free forever. No credit card."}
        </p>
      </motion.div>

      {mode === "signup" && (
        <FloatingInput
          id="name"
          label="Full name"
          value={name}
          onChange={setName}
          error={errors.name}
          autoComplete="name"
        />
      )}
      <FloatingInput
        id="email"
        label="Email"
        type="email"
        value={email}
        onChange={setEmail}
        error={errors.email}
        autoComplete="email"
      />
      <FloatingInput
        id="password"
        label="Password"
        type="password"
        value={password}
        onChange={setPassword}
        error={errors.password}
        autoComplete={mode === "login" ? "current-password" : "new-password"}
      />
      {mode === "signup" && (
        <FloatingInput
          id="confirm"
          label="Confirm password"
          type="password"
          value={confirm}
          onChange={setConfirm}
          error={errors.confirm}
          autoComplete="new-password"
        />
      )}

      {mode === "login" && (
        <div className="text-right">
          <Link to="/forgot-password" className="text-xs font-medium text-primary hover:underline">
            Forgot password?
          </Link>
        </div>
      )}

      <AnimatePresence>
        {formError && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <p className="text-xs text-destructive text-center px-2">{formError}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex justify-center pt-2">
        <SubmitButton loading={loading} label={mode === "login" ? "Sign in" : "Create account"} />
      </div>

      <div className="relative my-4">
        <div className="absolute inset-0 flex items-center" aria-hidden>
          <div className="w-full h-px bg-border" />
        </div>
        <div className="relative flex justify-center">
          <span className="px-3 bg-background text-[11px] uppercase tracking-wider text-muted-foreground">
            or
          </span>
        </div>
      </div>

      <GoogleButton onError={setFormError} />

      <p className="text-center text-sm text-muted-foreground pt-4">
        {mode === "login" ? (
          <>
            New here?{" "}
            <Link to="/signup" className="text-primary font-semibold hover:underline">
              Create an account
            </Link>
          </>
        ) : (
          <>
            Already have an account?{" "}
            <Link to="/login" className="text-primary font-semibold hover:underline">
              Sign in
            </Link>
          </>
        )}
      </p>
    </form>
  );
}

function AuthShell({ mode }: { mode: Mode }) {
  const { session, loading } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    if (!loading && session) navigate("/trips", { replace: true });
  }, [loading, session, navigate]);

  return (
    <div className="min-h-screen grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] bg-background">
      <IllustrationPanel mode={mode} />
      <section className="relative flex flex-col overflow-hidden">
        {/* subtle floating accents on the light side */}
        <div aria-hidden className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-16 -right-16 w-56 h-56 rounded-full bg-primary/5 blur-3xl" />
          <div className="absolute bottom-0 left-0 w-48 h-48 rounded-full bg-primary/[0.04] blur-3xl" />
          {FLOAT_SHAPES_AUTH.slice(0, 3).map((s, i) => (
            <FloatShape key={i} s={{ ...s, color: "hsl(var(--primary))" }} i={i} />
          ))}
        </div>

        <motion.header
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="relative lg:hidden px-6 h-[72px] flex items-center"
        >
          <Link to="/" className="inline-flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-primary flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-display font-bold text-primary text-[22px] tracking-tight">
              Wayfinder
            </span>
          </Link>
        </motion.header>

        <div className="relative flex-1 flex items-center justify-center px-6 py-10">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="w-full"
          >
            <AuthForm mode={mode} />
          </motion.div>
        </div>
      </section>
    </div>
  );
}

export function Login() {
  return <AuthShell mode="login" />;
}

export function Signup() {
  return <AuthShell mode="signup" />;
}
