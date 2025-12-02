import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import { Home, Search, AlertTriangle, ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="min-h-screen bg-white text-slate-900 flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center max-w-md w-full px-4"
      >
        {/* 404 Animation */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="mb-8 relative"
        >
          <div className="flex items-center justify-center">
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
              className="text-8xl font-bold text-slate-200"
            >
              4
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.3 }}
              className="mx-2"
            >
              <AlertTriangle className="h-12 w-12 text-orange-500" />
            </motion.div>
            <motion.div
              animate={{ rotate: [0, -10, 10, 0] }}
              transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
              className="text-8xl font-bold text-slate-200"
            >
              4
            </motion.div>
          </div>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.3 }}
          className="text-3xl font-bold text-slate-900 mb-4"
        >
          Page Not Found
        </motion.h1>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.3 }}
          className="text-lg text-slate-600 mb-8"
        >
          Oops! The page you're looking for doesn't exist or has been moved.
        </motion.p>

        {/* Search Suggestions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.3 }}
          className="mb-12 w-full"
        >
          <div className="text-sm text-slate-500 mb-3">You might be looking for:</div>
          <div className="space-y-2">
            <Link to="/" className="block p-3 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-colors">
              <div className="flex items-center gap-3">
                <Home className="h-4 w-4 text-slate-400" />
                <span className="text-sm text-slate-700">Home - PH5 Detector</span>
              </div>
            </Link>
            <Link to="/docs" className="block p-3 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-colors">
              <div className="flex items-center gap-3">
                <Search className="h-4 w-4 text-slate-400" />
                <span className="text-sm text-slate-700">Documentation</span>
              </div>
            </Link>
            <Link to="/playground" className="block p-3 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-colors">
              <div className="flex items-center gap-3">
                <AlertTriangle className="h-4 w-4 text-slate-400" />
                <span className="text-sm text-slate-700">Playground (Under Development)</span>
              </div>
            </Link>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.3 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <Link to="/">
            <Button className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-3 font-medium shadow-lg hover:shadow-xl transition-all duration-200">
              Go Home
            </Button>
          </Link>
          <Button 
            variant="outline" 
            className="border-slate-300 text-slate-700 hover:bg-slate-50 px-6 py-3 font-medium"
            onClick={() => window.history.back()}
          >
            Go Back
          </Button>
        </motion.div>

        {/* Fun Animation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mt-12 text-xs text-slate-400"
        >
          Error Code: 404 | Page Missing in Action
        </motion.div>
      </motion.div>
    </div>
  );
}
