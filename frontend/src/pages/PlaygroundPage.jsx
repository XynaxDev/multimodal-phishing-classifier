import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import { ArrowLeft, Wrench, Clock, Zap } from "lucide-react";
import { Link } from "react-router-dom";
import SiteFooter from "../components/SiteFooter";

export default function PlaygroundPage() {
  return (
    <div className="min-h-screen bg-white text-slate-900">
      <div className="relative z-10 flex min-h-screen flex-col">
        {/* Navigation */}
        <nav className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/80 backdrop-blur">
          <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3 md:py-4">
            <Link to="/" className="flex items-center gap-3">
              <ArrowLeft className="h-4 w-4 text-slate-600" />
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-orange-400 to-orange-600 shadow-lg shadow-orange-200">
                <span className="text-white font-black text-lg">P5</span>
              </div>
              <div className="leading-tight">
                <div className="text-xl font-black text-slate-900 tracking-tight">PH5</div>
              </div>
            </Link>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1">
          <div className="mx-auto flex w-full max-w-3xl flex-col px-4 pb-16 pt-8 md:pb-20">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col items-center text-center py-20"
            >
              {/* Icon */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1, duration: 0.3 }}
                className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-orange-100"
              >
                <Wrench className="h-10 w-10 text-orange-500" />
              </motion.div>

              {/* Title */}
              <motion.h1
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.3 }}
                className="text-3xl font-bold text-slate-900 mb-4"
              >
                Playground Under Development
              </motion.h1>

              {/* Description */}
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.3 }}
                className="text-lg text-slate-600 mb-8 max-w-md"
              >
                We're building an advanced playground for testing and experimenting with PH5's multimodal phishing detection capabilities.
              </motion.p>

              {/* Features Grid */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.3 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12 w-full max-w-2xl"
              >
                <div className="flex flex-col items-center p-6 rounded-xl border border-slate-200 bg-slate-50">
                  <Clock className="h-8 w-8 text-orange-500 mb-3" />
                  <h3 className="font-semibold text-slate-900 mb-2">Coming Soon</h3>
                  <p className="text-sm text-slate-600 text-center">Advanced testing tools and scenarios</p>
                </div>
                <div className="flex flex-col items-center p-6 rounded-xl border border-slate-200 bg-slate-50">
                  <Zap className="h-8 w-8 text-orange-500 mb-3" />
                  <h3 className="font-semibold text-slate-900 mb-2">Interactive</h3>
                  <p className="text-sm text-slate-600 text-center">Real-time detection and analysis</p>
                </div>
                <div className="flex flex-col items-center p-6 rounded-xl border border-slate-200 bg-slate-50">
                  <Wrench className="h-8 w-8 text-orange-500 mb-3" />
                  <h3 className="font-semibold text-slate-900 mb-2">Customizable</h3>
                  <p className="text-sm text-slate-600 text-center">Tailor detection parameters</p>
                </div>
              </motion.div>

              {/* CTA Button */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.3 }}
              >
                <Link to="/">
                  <Button className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 text-base font-medium shadow-lg hover:shadow-xl transition-all duration-200">
                    Back to Home
                  </Button>
                </Link>
              </motion.div>

              {/* Progress Indicator */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6, duration: 0.5 }}
                className="mt-12 w-full max-w-md"
              >
                <div className="text-sm text-slate-500 mb-2">Development Progress</div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: "65%" }}
                    transition={{ delay: 0.7, duration: 1, ease: "easeOut" }}
                    className="bg-orange-500 h-2 rounded-full"
                  />
                </div>
                <div className="text-xs text-slate-400 mt-1">65% Complete</div>
              </motion.div>
            </motion.div>
          </div>
        </main>

        {/* Footer */}
        <SiteFooter />
      </div>
    </div>
  );
}
