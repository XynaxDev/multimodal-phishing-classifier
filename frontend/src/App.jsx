import React, { useState } from "react";
import { Routes, Route } from "react-router-dom";
import NavBar from "./components/NavBar";
import AnnouncementBanner from "./components/AnnouncementBanner";
import PremiumHero from "./components/PremiumHero";
import PremiumSearchBar from "./components/PremiumSearchBar";
import InferencePanel from "./components/InferencePanel";
import FaqSection from "./components/FaqSection";
import SiteFooter from "./components/SiteFooter";
import { textCheck, urlCheck, imageCheckFile, chatMultipart } from "./api";
import PlaygroundPage from "./pages/PlaygroundPage";
import DocsPage from "./pages/DocsPage";
import NotFoundPage from "./pages/NotFoundPage";

// Home Page Component
function HomePage() {
  const [mode, setMode] = useState("text");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  async function run() {
    setIsLoading(true);
    setResult(null);

    try {
      let response;
      if (mode === "urls") {
        response = await urlCheck(query);
      } else if (mode === "images") {
        if (!file) {
          alert("Upload image first!");
          return;
        }
        response = await imageCheckFile(file);
      } else if (mode === "chat") {
        response = await chatMultipart(query, file);
      } else if (mode === "script") {
        response = await textCheck(query);
      } else {
        response = await textCheck(query);
      }
      
      console.log("API Response:", response);
      console.log("Response Data:", response.data);
      setResult(response.data);
    } catch (err) {
      console.error("API Error:", err);
      console.error("Error Response:", err?.response?.data);
      setResult({ error: err?.response?.data || err.message });
    } finally {
      setIsLoading(false);
    }
  }

  const clearResults = () => {
    setResult(null);
    setQuery("");
    setFile(null);
  };

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <div className="relative z-10 flex min-h-screen flex-col">
        <NavBar />

        <main className="flex-1">
          <div className="mx-auto flex w-full max-w-3xl flex-col px-4 pb-16 pt-8 md:pb-20">
            <AnnouncementBanner />
            <PremiumHero />

            <section id="modes" className="mt-6">
              <PremiumSearchBar
                baseQuery={query}
                setBaseQuery={setQuery}
                mode={mode}
                setMode={setMode}
                baseFile={file}
                setBaseFile={setFile}
                run={run}
                loading={isLoading}
                onClearResults={clearResults}
              />
            </section>

            <section className="mt-10">
              <InferencePanel data={result} loading={isLoading} mode={mode} onClear={clearResults} />
            </section>

            <FaqSection />
          </div>
        </main>

        <SiteFooter />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/playground" element={<PlaygroundPage />} />
      <Route path="/docs" element={<DocsPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
