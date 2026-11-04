import { useId } from "react";

interface SparklineProps {
  points: number[];
  color?: string;
  height?: number;
  width?: number | string;
  showGradient?: boolean;
  showDot?: boolean;
  className?: string;
}

export default function Sparkline({
  points,
  color,
  height = 32,
  width = "100%",
  showGradient = true,
  showDot = true,
  className = "",
}: SparklineProps) {
  const gradientId = useId();

  if (!points || points.length < 2) {
    return <div style={{ height }} className={`w-full opacity-20 ${className}`} />;
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const W = 100;
  const H = height;
  const pad = 4;
  const usableH = H - pad * 2;

  const pts = points.map((v, i) => {
    const x = (i / (points.length - 1)) * W;
    const y = H - pad - ((v - min) / range) * usableH;
    return { x, y };
  });

  const isUp = points[points.length - 1] >= points[0];
  const strokeColor = color || (isUp ? "#10B981" : "#F43F5E");

  // Build smooth cubic bezier curve
  let curvePath = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const mx = (p0.x + p1.x) / 2;
    curvePath += ` C ${mx.toFixed(1)} ${p0.y.toFixed(1)}, ${mx.toFixed(1)} ${p1.y.toFixed(1)}, ${p1.x.toFixed(1)} ${p1.y.toFixed(1)}`;
  }

  const areaPath = `${curvePath} L ${W} ${H} L 0 ${H} Z`;
  const lastPt = pts[pts.length - 1];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      style={{ height, width }}
      className={`overflow-visible ${className}`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity={0.28} />
          <stop offset="100%" stopColor={strokeColor} stopOpacity={0.0} />
        </linearGradient>
      </defs>

      {showGradient && (
        <path d={areaPath} fill={`url(#${gradientId})`} />
      )}

      <path
        d={curvePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />

      {showDot && (
        <>
          <circle
            cx={lastPt.x}
            cy={lastPt.y}
            r="4.5"
            fill={strokeColor}
            opacity="0.3"
          />
          <circle
            cx={lastPt.x}
            cy={lastPt.y}
            r="2"
            fill={strokeColor}
          />
        </>
      )}
    </svg>
  );
}

