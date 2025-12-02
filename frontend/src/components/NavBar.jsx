import { Github } from "lucide-react";
import { Link } from "react-router-dom";

export default function NavBar() {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/80 backdrop-blur">
      <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3 md:py-4">
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-orange-400 to-orange-600 shadow-lg shadow-orange-200">
              <span className="text-white font-black text-lg">P5</span>
            </div>
            <div className="leading-tight">
              <div className="text-xl font-black text-slate-900 tracking-tight">PH5</div>
            </div>
          </Link>
        </div>

        <nav className="flex items-center gap-4 text-xs text-slate-500 md:text-sm">
          <Link to="/playground" className="hidden rounded-full px-3 py-1 hover:bg-slate-100 md:inline-block">
            Playground
          </Link>
          <Link to="/docs" className="hidden rounded-full px-3 py-1 hover:bg-slate-100 md:inline-block">
            Docs
          </Link>
          <a
            href="https://github.com/XynaxDev/multimodal-phishing-detector"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 shadow-sm hover:border-slate-300 hover:bg-slate-50"
          >
            <Github className="h-4 w-4" />
            <span className="hidden md:inline">View on GitHub</span>
          </a>
        </nav>
      </div>
    </header>
  );
}
