import type { CSSProperties } from 'react';
import {
  Activity, ArrowRight, Ban, Bot, Check, ChevronDown, ChevronRight, ChevronUp, ChevronsUpDown,
  CircleCheck, CircleDashed, CircleDot, CircleMinus, CirclePause, CircleSlash, Clock, Copy,
  Download, Eye, FileCog, FileSpreadsheet, FlaskConical, FolderOpen, FolderPlus, GitMerge,
  History, Inbox, Info, Library, ListFilter, ListTree, LoaderCircle, Minus, OctagonAlert,
  OctagonX, PenLine, PencilRuler, Play, Plus, RefreshCw, Replace, RotateCw, Search, Shapes,
  Square, TestTube, TriangleAlert, Unplug, Upload, UserRound, X,
  type LucideIcon,
} from 'lucide-react';

/**
 * The glyphs the system uses, by Lucide name. Meanings are fixed (see the design
 * guide's ICONOGRAPHY): add an icon here rather than importing Lucide elsewhere.
 */
const ICONS = {
  Activity, ArrowRight, Ban, Bot, Check, ChevronDown, ChevronRight, ChevronUp, ChevronsUpDown,
  CircleCheck, CircleDashed, CircleDot, CircleMinus, CirclePause, CircleSlash, Clock, Copy,
  Download, Eye, FileCog, FileSpreadsheet, FlaskConical, FolderOpen, FolderPlus, GitMerge,
  History, Inbox, Info, Library, ListFilter, ListTree, LoaderCircle, Minus, OctagonAlert,
  OctagonX, PenLine, PencilRuler, Play, Plus, RefreshCw, Replace, RotateCw, Search, Shapes,
  Square, TestTube, TriangleAlert, Unplug, Upload, UserRound, X,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICONS;

export interface IconProps {
  name: IconName;
  /** Pixel size. 16 in the app, 14 in badges. */
  size?: number;
  strokeWidth?: number;
  color?: string;
  /** Accessible name; without it the icon is decorative (aria-hidden). */
  title?: string;
  className?: string;
  style?: CSSProperties;
}

/** One Lucide glyph with the system's size and stroke contract. */
export function Icon({ name, size = 16, strokeWidth = 1.75, color = 'currentColor', title, className, style }: IconProps) {
  const Glyph = ICONS[name];
  return (
    <Glyph
      size={size}
      strokeWidth={strokeWidth}
      color={color}
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      className={className}
      style={{ flexShrink: 0, display: 'inline-block', verticalAlign: 'middle', ...style }}
    />
  );
}
