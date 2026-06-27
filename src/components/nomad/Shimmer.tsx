import { cn } from "@/lib/utils";

/** Skeleton block with left-to-right shimmer sweep. */
export function Shimmer({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-md bg-muted",
        "before:absolute before:inset-0 before:-translate-x-full",
        "before:bg-gradient-to-r before:from-transparent before:via-foreground/10 before:to-transparent",
        "before:animate-[shimmer_1.5s_infinite]",
        className
      )}
      aria-hidden="true"
    />
  );
}
