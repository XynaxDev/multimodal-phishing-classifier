import { textCheck } from "../api";
import { useState } from "react";

export default function TextCheck() {
  const [text, setText] = useState("");
  const [out, setOut] = useState(null);

  async function run() {
    const r = await textCheck(text);
    setOut(r);
  }

  return (
    <div>
      <textarea value={text} onChange={e=>setText(e.target.value)} placeholder="Paste email or message…" />
      <button onClick={run}>Analyze</button>

      {out && <pre>{JSON.stringify(out, null, 2)}</pre>}
    </div>
  );
}
