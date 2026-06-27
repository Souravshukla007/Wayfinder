import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-6 py-20 text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="relative w-full max-w-md"
      >
        <motion.div
          className="pointer-events-none flex justify-center text-primary/90"
          animate={{ y: [0, -14, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          aria-hidden="true"
        >
          <svg
            width="280"
            height="160"
            viewBox="0 0 320 180"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="w-full max-w-[280px]"
          >
            {/* Clouds */}
            <ellipse cx="48" cy="52" rx="24" ry="12" fill="currentColor" opacity="0.12" />
            <ellipse cx="70" cy="58" rx="18" ry="10" fill="currentColor" opacity="0.10" />
            <ellipse cx="260" cy="46" rx="22" ry="11" fill="currentColor" opacity="0.12" />
            <ellipse cx="282" cy="52" rx="16" ry="9" fill="currentColor" opacity="0.10" />

            {/* Globe */}
            <circle cx="160" cy="112" r="44" stroke="currentColor" strokeWidth="3" opacity="0.18" />
            <path
              d="M132 112C132 95.43 144.53 82 160 82C175.47 82 188 95.43 188 112"
              stroke="currentColor"
              strokeWidth="2"
              opacity="0.14"
            />
            <path
              d="M132 112C132 128.57 144.53 142 160 142C175.47 142 188 128.57 188 112"
              stroke="currentColor"
              strokeWidth="2"
              opacity="0.14"
            />
            <line x1="160" y1="68" x2="160" y2="156" stroke="currentColor" strokeWidth="2" opacity="0.14" />
            <line x1="116" y1="112" x2="204" y2="112" stroke="currentColor" strokeWidth="2" opacity="0.14" />

            {/* Dotted flight path */}
            <path
              d="M52 96C92 76 120 120 160 108C200 96 232 64 272 84"
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray="6 6"
              opacity="0.35"
            />

            {/* Paper plane */}
            <g transform="translate(264 78) rotate(12)">
              <path
                d="M0 0L18 6L6 10L0 22L-6 10L-18 6L0 0Z"
                fill="currentColor"
              />
            </g>

            {/* Map pin */}
            <g transform="translate(160 68)">
              <path
                d="M0 0C-7.73 0-14 6.27-14 14C-14 24.5 0 38 0 38C0 38 14 24.5 14 14C14 6.27 7.73 0 0 0Z"
                fill="currentColor"
                opacity="0.85"
              />
              <circle cx="0" cy="14" r="5" fill="hsl(var(--background))" />
            </g>
          </svg>
        </motion.div>

        <h1 className="font-display relative z-10 -mt-6 text-[6.5rem] font-bold leading-none tracking-[-0.04em] text-foreground sm:text-[8rem]">
          404
        </h1>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 mt-4 max-w-xs"
      >
        <p className="text-lg leading-relaxed text-muted-foreground">
          Looks like this destination doesn't exist.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 mt-8"
      >
        <Button
          onClick={() => navigate("/")}
          size="lg"
          className="rounded-full px-8 text-base"
        >
          <ArrowLeft className="size-4" />
          Back to home
        </Button>
      </motion.div>
    </div>
  );
};

export default NotFound;
