import { chatMultipart } from "../api";
import { useState } from "react";

export default function ChatMode() {
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [out, setOut] = useState(null);

  async function run() {
    const r = await chatMultipart(query, file);
    setOut(r);
  }

  return (
    <div>
      <textarea value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ask anything… (you can also upload an image)" />
      <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])} />

      <button onClick={run}>Send</button>

      {out && <pre>{JSON.stringify(out, null, 2)}</pre>}
    </div>
  );
}
