import React, { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, Shield, Zap, Globe, Brain, Lock, Sparkles, ArrowRight } from "lucide-react";

const faqs = [
  {
    q: "What types of content can PH5 analyze?",
    a: "PH5 can analyze text messages, emails, URLs, chat conversations, and screenshots of potential phishing content. Simply upload an image or paste raw text to get an instant verdict with confidence scores.",
  },
  {
    q: "How accurate is PH5's phishing detection?",
    a: "PH5 uses advanced multimodal AI models trained on millions of phishing examples. Our system achieves over 95% accuracy in detecting sophisticated phishing attempts, with real-time analysis and confidence scoring.",
  },
  {
    q: "Does PH5 store my data or protect my privacy?",
    a: "Your privacy is our priority. All content is processed in-memory for inference only and is never persisted to databases or shared with third parties. We don't store or analyze your personal data.",
  },
  {
    q: "Can I integrate PH5 into my own application?",
    a: "Yes! PH5 provides a comprehensive REST API that you can integrate into your applications. The same backend endpoints powering this UI are available for developers to build custom solutions.",
  },
  {
    q: "What makes PH5 different from other phishing detectors?",
    a: "PH5 uses multimodal AI analysis, combining text, image, and URL analysis in real-time. Unlike traditional tools that check single data types, PH5 provides comprehensive detection across all communication channels.",
  },
  {
    q: "Is PH5 suitable for enterprise use?",
    a: "Absolutely. PH5 is designed for both individual users and enterprise deployments. We offer scalable solutions with batch processing, API access, and customizable detection parameters for business needs.",
  },
];

const features = [
  {
    icon: <Brain className="h-6 w-6" />,
    title: "AI-Powered Analysis",
    description: "Advanced machine learning models for intelligent phishing detection"
  },
  {
    icon: <Zap className="h-6 w-6" />,
    title: "Real-Time Results",
    description: "Get instant analysis with sub-second response times"
  },
  {
    icon: <Shield className="h-6 w-6" />,
    title: "Multimodal Detection",
    description: "Analyze text, images, URLs, and chat simultaneously"
  },
  {
    icon: <Lock className="h-6 w-6" />,
    title: "Privacy First",
    description: "Your data is processed in-memory with zero storage"
  },
  {
    icon: <Globe className="h-6 w-6" />,
    title: "Global Coverage",
    description: "Detect phishing attempts in multiple languages and formats"
  },
  {
    icon: <Sparkles className="h-6 w-6" />,
    title: "Smart Scoring",
    description: "Confidence scores and detailed threat analysis"
  }
];

function FeaturesSection() {
  return (
    <section className="relative mt-20 overflow-hidden">
      {/* Geometric Background */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-0 left-0 w-64 h-64 bg-orange-500 rounded-full blur-3xl" />
        <div className="absolute top-20 right-0 w-96 h-96 bg-orange-400 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/2 w-80 h-80 bg-purple-500 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-0 w-72 h-72 bg-orange-300 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-1/3 w-88 h-88 bg-red-500 rounded-full blur-3xl" />
      </div>
      
      {/* Geometric Pattern */}
      <div className="absolute inset-0">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-slate-200" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="relative max-w-3xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-4">
            Why Choose <span className="text-orange-500">PH5</span>?
          </h2>
          <p className="text-base sm:text-lg text-slate-600 max-w-2xl mx-auto px-4">
            Experience the future of phishing detection with our cutting-edge multimodal AI technology
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1, duration: 0.3 }}
              viewport={{ once: true }}
              className="group"
            >
              <div className="h-full p-4 sm:p-6 rounded-2xl border border-slate-200 bg-white/50 backdrop-blur-sm hover:bg-white hover:shadow-lg hover:border-orange-200 transition-all duration-300">
                <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-orange-100 text-orange-600 mb-4 group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
                <h3 className="text-base sm:text-lg font-semibold text-slate-900 mb-2">{feature.title}</h3>
                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">{feature.description}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.3 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-orange-100 text-orange-700 font-medium">
            <Sparkles className="h-4 w-4" />
            <span>Powered by Advanced AI Technology</span>
            <ArrowRight className="h-4 w-4" />
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default function FaqSection() {
  const [openIndex, setOpenIndex] = useState(null);

  return (
    <>
      <FeaturesSection />
      
      <section className="relative mt-20 mb-12">
        {/* Background Graphics */}
        <div className="absolute inset-0 opacity-3">
          <div className="absolute top-0 left-1/4 w-32 h-32 bg-orange-400 rounded-full blur-2xl" />
          <div className="absolute bottom-0 right-1/4 w-40 h-40 bg-orange-300 rounded-full blur-2xl" />
        </div>

        <div className="relative max-w-3xl mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-4">
              Frequently Asked <span className="text-orange-500">Questions</span>
            </h2>
            <p className="text-base sm:text-lg text-slate-600 px-4">
              Everything you need to know about PH5's phishing detection capabilities
            </p>
          </motion.div>

          <div className="space-y-4">
            {faqs.map((item, index) => {
              const isOpen = openIndex === index;
              return (
                <motion.div
                  key={item.q}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1, duration: 0.3 }}
                  viewport={{ once: true }}
                >
                  <button
                    type="button"
                    onClick={() => setOpenIndex(isOpen ? null : index)}
                    className="w-full text-left group"
                  >
                    <div className="p-4 sm:p-6 rounded-2xl border border-slate-200 bg-white hover:border-orange-200 hover:shadow-md transition-all duration-300">
                      <div className="flex items-center justify-between gap-2 sm:gap-4">
                        <p className="font-semibold text-slate-900 text-sm sm:text-base pr-2">{item.q}</p>
                        <motion.div
                          animate={{ rotate: isOpen ? 180 : 0 }}
                          transition={{ duration: 0.2 }}
                          className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center group-hover:bg-orange-200 transition-colors"
                        >
                          <ChevronDown className="h-3 w-3 sm:h-4 sm:w-4" />
                        </motion.div>
                      </div>
                      
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ 
                          height: isOpen ? "auto" : 0, 
                          opacity: isOpen ? 1 : 0 
                        }}
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                        className="overflow-hidden"
                      >
                        <p className="pt-3 sm:pt-4 text-xs sm:text-sm text-slate-600 leading-relaxed border-t border-slate-100 mt-3 sm:mt-4">
                          {item.a}
                        </p>
                      </motion.div>
                    </div>
                  </button>
                </motion.div>
              );
            })}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.3 }}
            viewport={{ once: true }}
            className="text-center mt-12"
          >
            <p className="text-sm sm:text-base text-slate-600 mb-4">Still have questions?</p>
            <a
              href="/docs"
              className="inline-flex items-center gap-2 px-4 sm:px-6 py-2 sm:py-3 rounded-xl bg-orange-500 hover:bg-orange-600 text-white font-medium shadow-lg hover:shadow-xl transition-all duration-200 text-sm sm:text-base"
            >
              Visit Documentation
              <ArrowRight className="h-3 w-3 sm:h-4 sm:w-4" />
            </a>
          </motion.div>
        </div>
      </section>
    </>
  );
}
