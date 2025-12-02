import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE || "/api";

const instance = axios.create({
  baseURL: BASE,
  timeout: 60_000,
});

export async function textCheck(text) {
  return instance.post("/text_check", { text });
}
export async function urlCheck(url) {
  return instance.post("/url_check", { url });
}
export async function imageCheckFile(file) {
  const form = new FormData();
  form.append("image_file", file);
  return instance.post("/image_check", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}
export async function chatMultipart(query, file) {
  const form = new FormData();
  form.append("query", query);
  if (file) form.append("file", file);
  return instance.post("/chat_multipart", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}
