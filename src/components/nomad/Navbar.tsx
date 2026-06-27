import { useEffect, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { LogOut, Menu, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";

function initialsFromEmail(email?: string | null) {
  if (!email) return "U";
  const handle = email.split("@")[0] ?? "";
  return (handle.slice(0, 2) || "U").toUpperCase();
}

function UserMenu({ variant = "desktop" }: { variant?: "desktop" | "mobile" }) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const email = user?.email ?? "";
  const initials = initialsFromEmail(email);

  const onSignOut = async () => {
    await signOut();
    navigate("/");
  };

  if (variant === "mobile") {
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-3 px-1">
          <Avatar className="h-9 w-9">
            <AvatarFallback className="bg-primary text-primary-foreground text-xs font-semibold">
              {initials}
            </AvatarFallback>
          </Avatar>
          <span className="text-sm text-muted-foreground truncate">{email}</span>
        </div>
        <Button variant="ghost" className="rounded-full justify-center gap-2" onClick={onSignOut}>
          <LogOut className="h-4 w-4" /> Sign out
        </Button>
      </div>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <Avatar className="h-9 w-9">
          <AvatarFallback className="bg-primary text-primary-foreground text-xs font-semibold">
            {initials}
          </AvatarFallback>
        </Avatar>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel className="truncate font-normal text-muted-foreground">
          {email}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => navigate("/trips")}>My Trips</DropdownMenuItem>
        <DropdownMenuItem onClick={() => navigate("/preferences")}>Preferences</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onSignOut} className="gap-2 text-destructive focus:text-destructive">
          <LogOut className="h-4 w-4" /> Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/plan", label: "Plan" },
  { to: "/trips", label: "My Trips" },
  { to: "/preferences", label: "Preferences" },
];

function NavItem({ to, label, onClick }: { to: string; label: string; onClick?: () => void }) {
  const { pathname } = useLocation();
  const active = to === "/" ? pathname === "/" : pathname.startsWith(to);
  return (
    <NavLink
      to={to}
      onClick={onClick}
      className="relative px-3 py-2 text-sm font-medium text-foreground/80 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-full"
    >
      <span>{label}</span>
      <span
        aria-hidden
        className={cn(
          "absolute left-1/2 -bottom-0.5 h-0.5 -translate-x-1/2 rounded-full bg-primary transition-[width] duration-200 ease-out",
          active ? "w-[60%]" : "w-0 group-hover:w-[60%]"
        )}
      />
    </NavLink>
  );
}

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  const { session } = useAuth();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => setOpen(false), [pathname]);

  return (
    <>
      <header
        className={cn(
          "fixed inset-x-0 top-0 z-50 transition-all duration-300",
          scrolled
            ? "backdrop-blur-xl bg-background/70 shadow-[0_1px_0_0_hsl(var(--border)/0.6)]"
            : "bg-transparent"
        )}
      >
        <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-2 font-display text-lg font-semibold tracking-tight">
            <span className="grid h-8 w-8 place-items-center rounded-full bg-primary text-primary-foreground">
              <Sparkles className="h-4 w-4" />
            </span>
            <span className="text-primary">Wayfinder</span>
          </Link>

          <div className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map((l) => (
              <NavItem key={l.to} {...l} />
            ))}
          </div>

          <div className="hidden md:flex items-center gap-2">
            {session ? (
              <UserMenu />
            ) : (
              <>
                <Button asChild variant="ghost" className="rounded-full">
                  <Link to="/login">Log in</Link>
                </Button>
                <Button asChild className="rounded-full px-5">
                  <Link to="/signup">Sign up</Link>
                </Button>
              </>
            )}
          </div>

          <button
            type="button"
            className="md:hidden inline-flex h-11 w-11 items-center justify-center rounded-full hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Open menu"
            aria-expanded={open}
            onClick={() => setOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </button>
        </nav>
      </header>

      <AnimatePresence>
        {open && (
          <>
            <motion.div
              key="overlay"
              className="fixed inset-0 z-[60] bg-foreground/40 backdrop-blur-sm md:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setOpen(false)}
            />
            <motion.aside
              key="drawer"
              className="fixed right-0 top-0 z-[61] h-dvh w-80 max-w-[85vw] bg-background p-6 shadow-2xl md:hidden"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ duration: 0.3, ease: [0.32, 0.72, 0, 1] }}
              role="dialog"
              aria-modal="true"
              aria-label="Mobile menu"
            >
              <div className="flex items-center justify-between">
                <span className="font-display text-lg font-semibold">Menu</span>
                <button
                  type="button"
                  className="inline-flex h-11 w-11 items-center justify-center rounded-full hover:bg-muted"
                  aria-label="Close menu"
                  onClick={() => setOpen(false)}
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <nav className="mt-8 flex flex-col gap-1">
                {NAV_LINKS.map((l) => (
                  <NavLink
                    key={l.to}
                    to={l.to}
                    onClick={() => setOpen(false)}
                    className={({ isActive }) =>
                      cn(
                        "rounded-full px-4 py-3 text-base font-medium transition-colors",
                        isActive ? "bg-primary/10 text-primary" : "hover:bg-muted"
                      )
                    }
                  >
                    {l.label}
                  </NavLink>
                ))}
              </nav>

              <div className="mt-8 flex flex-col gap-2">
                {session ? (
                  <UserMenu variant="mobile" />
                ) : (
                  <>
                    <Button asChild variant="ghost" className="rounded-full justify-center">
                      <Link to="/login">Log in</Link>
                    </Button>
                    <Button asChild className="rounded-full justify-center">
                      <Link to="/signup">Sign up</Link>
                    </Button>
                  </>
                )}
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
