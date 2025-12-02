import { imageCheckFile } from "../api";
import { useState } from "react";

export default function ImageCheck() {
  const [file, setFile] = useState(null);
  const [out, setOut] = useState(null);

  async function run() {
    const r = await imageCheckFile(file);
    setOut(r);
  }

  return (
    <div>
      <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])} />
      <button onClick={run}>Analyze Image</button>

      {out && <pre>{JSON.stringify(out, null, 2)}</pre>}
    </div>
  );
}
