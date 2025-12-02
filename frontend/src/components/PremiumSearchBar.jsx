import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Paperclip,
  Loader2,
  X,
  Globe,
  Mail,
  Smartphone,
  Code,
  Type,
  MessageSquare,
  Link,
  Image,
  MessageCircle,
  Bot,
} from "lucide-react";
import { Button } from "./ui/button";
import { useState, useEffect } from "react";

export default function PremiumSearchBar({
  baseQuery,
  setBaseQuery,
  mode,
  setMode,
  baseFile,
  setBaseFile,
  run,
  loading,
  onClearResults,
}) {
  // 🔑 ISOLATED STATE PER MODE
  const [modeQueries, setModeQueries] = useState({
    text: "",
    sms: "",
    emails: "",
    urls: "",
    images: "",
    script: "",
    chat: "",
  });

  const [modeFiles, setModeFiles] = useState({
    text: null,
    sms: null,
    emails: null,
    urls: null,
    images: null,
    script: null,
    chat: null,
  });

  const query = modeQueries[mode] || "";
  const file = modeFiles[mode] || null;

  const setQuery = (value) => {
    setModeQueries((prev) => ({ ...prev, [mode]: value }));
    setBaseQuery(value);
  };

  const setFile = (value) => {
    setModeFiles((prev) => ({ ...prev, [mode]: value }));
    setBaseFile(value);
  };

  // Sync internal state with props when they change (for clear results)
  useEffect(() => {
    if (!baseQuery) {
      setModeQueries((prev) => ({ ...prev, [mode]: "" }));
    }
  }, [baseQuery, mode]);

  useEffect(() => {
    if (!baseFile) {
      setModeFiles((prev) => ({ ...prev, [mode]: null }));
    }
  }, [baseFile, mode]);

  const isImageMode = mode === "images";
  const isChatMode = mode === "chat";

  const placeholder =
    mode === "urls"
      ? "Enter any suspicious URL to check for phishing..."
      : mode === "emails"
      ? "Enter any email content to analyze..."
      : mode === "sms"
      ? "Enter any SMS message to check..."
      : mode === "script"
      ? "Paste any script or code to analyze..."
      : mode === "chat"
      ? "Ask anything about phishing, security, or upload content..."
      : mode === "text"
      ? "Enter any suspicious message or text..."
      : "Enter content to analyze...";

  const getModeIcon = () => {
    switch (mode) {
      case "urls":
        return <Globe className="h-4 w-4" />;
      case "emails":
        return <Mail className="h-4 w-4" />;
      case "sms":
        return <Smartphone className="h-4 w-4" />;
      case "script":
        return <Code className="h-4 w-4" />;
      case "chat":
        return <Bot className="h-5 w-5" />;
      default:
        return <Globe className="h-4 w-4" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative mt-4 w-full rounded-[28px] border border-gray-200 bg-white px-4 py-4 md:px-6 md:py-5"
    >
      {/* TABS - Text color orange when active, bottom shadow */}
      <div className="flex justify-center mb-5">
        <div className="inline-flex items-center gap-1 bg-gray-100 rounded-full p-1 shadow-md shadow-gray-200 overflow-x-auto max-w-full sm:max-w-none">
          <TabButton 
            label="Text" 
            active={mode === "text"} 
            onClick={() => setMode("text")} 
          />
          <TabButton 
            label="SMS" 
            active={mode === "sms"} 
            onClick={() => setMode("sms")} 
          />
          <TabButton 
            label="Emails" 
            active={mode === "emails"} 
            onClick={() => setMode("emails")} 
          />
          <TabButton 
            label="URLs" 
            active={mode === "urls"} 
            onClick={() => setMode("urls")} 
          />
          <TabButton 
            label="Script" 
            active={mode === "script"} 
            onClick={() => setMode("script")} 
          />
          <TabButton 
            label="Images" 
            active={mode === "images"} 
            onClick={() => setMode("images")} 
          />
          <TabButton 
            label="Chat" 
            active={mode === "chat"} 
            onClick={() => setMode("chat")} 
          />
        </div>
      </div>

      {/* Image Preview - ONLY Chat Mode */}
      <AnimatePresence>
        {file && isChatMode && (
          <motion.div
            initial={{ opacity: 0, height: 0, marginTop: 0 }}
            animate={{ opacity: 1, height: "auto", marginTop: 12 }}
            exit={{ opacity: 0, height: 0, marginTop: 0 }}
            transition={{ duration: 0.2 }}
            className="flex flex-wrap gap-3 mb-3"
          >
            <div className="relative inline-block">
              <div className="h-20 w-20 overflow-hidden rounded-xl border border-gray-200 bg-gray-50 shadow-sm">
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="h-full w-full object-cover"
                />
              </div>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                type="button"
                onClick={() => setFile(null)}
                className="absolute -top-2 -right-2 inline-flex h-5 w-5 items-center justify-center rounded-full bg-gray-800 text-white hover:bg-gray-900 transition-colors shadow-md"
                aria-label="Remove image"
              >
                <X className="h-3 w-3" />
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* MAIN INPUT CONTAINER */}
      <div className="mt-0">
        {!isImageMode ? (
          // Text-based modes - Send button INSIDE input wrapper
          <div className="flex-1">
            <div className={`flex items-start rounded-2xl bg-gray-50 px-4 border border-gray-200 shadow-sm gap-3 ${isChatMode ? "py-4" : "py-3.5"}`}>
              {/* Icon - Properly aligned */}
              <div className={`flex shrink-0 items-center justify-center rounded-full bg-gray-100 text-gray-500 ${isChatMode ? "h-9 w-9 -mt-1" : "h-8 w-8"}`}>
                {getModeIcon()}
              </div>

              {/* Input/Textarea */}
              {isChatMode ? (
                <textarea
                  rows={4}
                  placeholder={placeholder}
                  className="flex-1 resize-none bg-transparent text-base text-black placeholder:text-gray-500 border-0 outline-none focus:outline-none focus:ring-0 focus:border-0 shadow-none font-normal leading-relaxed max-h-32 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent mt-0.5"
                  style={{
                    scrollbarWidth: "thin",
                    scrollbarColor: "#CBD5E0 transparent"
                  }}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              ) : (
                <input
                  type="text"
                  placeholder={placeholder}
                  className="flex-1 bg-transparent text-base text-black placeholder:text-gray-500 border-0 outline-none focus:outline-none focus:ring-0 focus:border-0 shadow-none font-normal h-8"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              )}

              {/* Action Buttons - Inside input wrapper */}
              <div className={`flex items-center gap-2 flex-shrink-0 ${isChatMode ? "self-end" : ""}`}>
                {/* Attachment Button - Chat only */}
                {isChatMode && (
                  <label className="inline-flex h-9 w-9 cursor-pointer items-center justify-center rounded-full border border-gray-300 bg-white text-gray-400 hover:bg-gray-100 hover:text-gray-600 hover:border-gray-400 transition-colors">
                    <Paperclip className="h-4 w-4" />
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    />
                  </label>
                )}

                {/* Send Button */}
                <Button
                  size="icon"
                  className={`inline-flex items-center justify-center rounded-lg bg-slate-900 text-white hover:bg-slate-800 active:bg-slate-950 transition-colors flex-shrink-0 ${isChatMode ? "h-9 w-9" : "h-8 w-8"}`}
                  onClick={run}
                  disabled={loading}
                  aria-label="Send"
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          // Image mode
          <div className="flex w-full flex-col gap-3">
            <label className="relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 px-4 py-8 text-center transition-colors hover:bg-gray-100 hover:border-gray-400">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />

              {file ? (
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="mx-auto max-h-40 rounded object-contain"
                />
              ) : (
                <>
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full border-2 bg-white text-gray-500">
                    <Paperclip className="h-5 w-5" />
                  </div>
                  <p className="mb-1.5 text-base font-semibold text-black">
                    Drop your image here
                  </p>
                  <p className="text-sm text-gray-500">
                    PNG, JPG or GIF (max ~2MB)
                  </p>
                  <span className="mt-5 inline-flex items-center justify-center rounded-full border border-gray-300 bg-white px-5 py-2 text-sm font-medium text-gray-800 hover:bg-gray-50 transition-colors">
                    Select image
                  </span>
                </>
              )}
            </label>

            {file && (
              <div className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 text-sm text-gray-800 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="h-12 w-12 overflow-hidden rounded-lg bg-gray-100 border border-gray-200">
                    <img
                      src={URL.createObjectURL(file)}
                      alt={file.name}
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="flex flex-col text-left">
                    <span className="max-w-[160px] truncate font-medium text-base">
                      {file.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setFile(null)}
                    className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:bg-gray-50 hover:border-gray-400 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <Button
                    size="icon"
                    className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white hover:bg-slate-800 active:bg-slate-950 transition-colors"
                    onClick={run}
                    disabled={loading}
                  >
                    {loading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Custom scrollbar styles */}
      <style jsx>{`
        textarea::-webkit-scrollbar {
          width: 6px;
        }
        textarea::-webkit-scrollbar-track {
          background: transparent;
        }
        textarea::-webkit-scrollbar-thumb {
          background: #CBD5E0;
          border-radius: 3px;
        }
        textarea::-webkit-scrollbar-thumb:hover {
          background: #A0AEC0;
        }
      `}</style>
    </motion.div>
  );
}

// Tab Button Component - Using React Icons from lucide-react
function TabButton({ label, active, onClick }) {
  const getIcon = () => {
    switch (label) {
      case "Text":
        return <Type className="h-4 w-4" />;
      case "SMS":
        return <MessageSquare className="h-4 w-4" />;
      case "Emails":
        return <Mail className="h-4 w-4" />;
      case "URLs":
        return <Link className="h-4 w-4" />;
      case "Script":
        return <Code className="h-4 w-4" />;
      case "Images":
        return <Image className="h-4 w-4" />;
      case "Chat":
        return <MessageCircle className="h-4 w-4" />;
      default:
        return null;
    }
  };

  return (
    <button
      onClick={onClick}
      className={`px-2 sm:px-3 py-1.5 rounded-full text-xs sm:text-sm font-medium transition-all whitespace-nowrap flex items-center gap-1.5 flex-shrink-0 ${
        active
          ? "text-orange-500 bg-white shadow-md"
          : "bg-transparent text-gray-600 hover:text-gray-900"
      }`}
    >
      <span className={`${active ? "text-orange-500" : "text-gray-500"}`}>
        {getIcon()}
      </span>
      <span className="hidden xs:inline sm:inline">{label}</span>
    </button>
  );
}
