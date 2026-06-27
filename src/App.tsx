import { useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { ThemeProvider } from "next-themes";
import { AuthProvider } from "@/contexts/AuthContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { Navbar } from "@/components/nomad/Navbar";
import { PageTransition } from "@/components/nomad/PageTransition";

import Landing from "./pages/Landing";
import Plan from "./pages/Plan";
import Preferences from "./pages/Preferences";
import Trips, { TripDetail } from "./pages/Trips";
import { Login, Signup } from "./pages/AuthPages";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Routes that render the global Wayfinder navbar.
// Auth pages (/login, /signup) intentionally excluded — they have their own layout.
const NAVBAR_ROUTES = ["/", "/plan", "/preferences", "/trips"];

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

function AnimatedRoutes() {
  const location = useLocation();
  const showNavbar = NAVBAR_ROUTES.some((p) =>
    p === "/" ? location.pathname === "/" : location.pathname.startsWith(p)
  );

  return (
    <>
      {showNavbar && <Navbar />}
      <AnimatePresence mode="wait" initial={false}>
        <Routes location={location} key={location.pathname}>
          {/* Public */}
          <Route path="/" element={<PageTransition><Landing /></PageTransition>} />
          <Route path="/plan" element={<PageTransition><Plan /></PageTransition>} />
          <Route path="/login" element={<PageTransition><Login /></PageTransition>} />
          <Route path="/signup" element={<PageTransition><Signup /></PageTransition>} />

          {/* Authenticated (Supabase session required) */}
          <Route
            path="/preferences"
            element={
              <ProtectedRoute>
                <PageTransition><Preferences /></PageTransition>
              </ProtectedRoute>
            }
          />
          <Route
            path="/trips"
            element={
              <ProtectedRoute>
                <PageTransition><Trips /></PageTransition>
              </ProtectedRoute>
            }
          />
          <Route
            path="/trips/:id"
            element={
              <ProtectedRoute>
                <PageTransition><TripDetail /></PageTransition>
              </ProtectedRoute>
            }
          />

          <Route path="*" element={<PageTransition><NotFound /></PageTransition>} />
        </Routes>
      </AnimatePresence>
    </>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} storageKey="app-theme">
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <ScrollToTop />
            <AnimatedRoutes />
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
