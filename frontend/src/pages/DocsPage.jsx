import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import { ArrowLeft, Github, Shield, Zap, Globe } from "lucide-react";
import { Link } from "react-router-dom";
import SiteFooter from "../components/SiteFooter";

export default function DocsPage() {
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
          <div className="mx-auto flex w-full max-w-4xl px-4 pb-16 pt-8 md:pb-20">
            {/* Hero Section with CTA */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-16"
            >
              <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">
                Ready to Get Started?
              </h1>
              <p className="text-xl text-slate-600 mb-8 max-w-2xl mx-auto">
                Experience the power of multimodal phishing detection. Try PH5 now and protect yourself from online threats.
              </p>
              
              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
                <Link to="/">
                  <Button className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-4 font-medium text-lg shadow-lg hover:shadow-xl transition-all duration-200">
                    Try PH5 Now
                  </Button>
                </Link>
                <Button 
                  variant="outline" 
                  className="border-orange-500 text-orange-600 hover:bg-orange-50 px-8 py-4 font-medium text-lg transition-all duration-200"
                  onClick={() => window.open('https://github.com/XynaxDev', '_blank')}
                >
                  <Github className="h-5 w-5 mr-2" />
                  View Source
                </Button>
              </div>

              {/* Feature Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center justify-center w-12 h-12 bg-orange-100 rounded-lg mb-4 mx-auto">
                    <Zap className="h-6 w-6 text-orange-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">Fast Detection</h3>
                  <p className="text-slate-600">Real-time analysis with sub-second response times</p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center justify-center w-12 h-12 bg-orange-100 rounded-lg mb-4 mx-auto">
                    <Globe className="h-6 w-6 text-orange-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">Multimodal</h3>
                  <p className="text-slate-600">Supports text, URLs, images, and chat analysis</p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center justify-center w-12 h-12 bg-orange-100 rounded-lg mb-4 mx-auto">
                    <Shield className="h-6 w-6 text-orange-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">AI-Powered</h3>
                  <p className="text-slate-600">Advanced LLM integration for intelligent analysis</p>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </main>

        {/* Footer */}
        <SiteFooter />
      </div>
    </div>
  );
}
