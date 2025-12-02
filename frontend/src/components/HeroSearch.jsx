import React from "react";

const modes = [
  { id: "text", label: "Text" },
  { id: "sms", label: "SMS" },
  { id: "urls", label: "URLs" },
  { id: "emails", label: "Emails" },
  { id: "images", label: "Images" },
  { id: "chat", label: "Chat" },
];

export default function HeroSearch({ mode, setMode, query, setQuery, file, setFile, onRun, loading }) {
  return (
    <div className="search-card">
      <div className="search-row">
        <div className="search-left">
          <svg className="globe" width="20" height="20" viewBox="0 0 24 24">
            <path fill="#777" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
          </svg>
          <input
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={mode === "urls" ? "https://example.com or example.com" : mode === "images" ? "Upload or choose an image below" : "Paste text, URL or message here"}
          />
        </div>

        <div className="search-cta">
          <button className="chip" aria-hidden>
            {mode === "images" ? "Image" : mode.toUpperCase()}
          </button>
          <button className="arrow-btn" onClick={onRun} disabled={loading}>
            {loading ? "…" : <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#fff" d="M12 2l10 10-10 10-2-2 6-6H2v-4h14l-6-6z"/></svg>}
          </button>
        </div>
      </div>

      <div className="modes-row">
        {modes.map((m) => (
          <button
            key={m.id}
            className={`mode-pill ${mode === m.id ? "active" : ""}`}
            onClick={() => setMode(m.id)}
          >
            {m.label}
          </button>
        ))}

        {/* file uploader for images */}
        <div style={{ marginLeft: "auto" }}>
          <label className="upload-label">
            <input
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <span className="upload-text">{file ? file.name : "Upload image"}</span>
          </label>
        </div>
      </div>
    </div>
  );
}
