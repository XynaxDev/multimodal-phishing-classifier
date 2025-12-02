import { Github, Instagram, Linkedin } from "lucide-react";

export default function SiteFooter() {
  const year = new Date().getFullYear();

  return (
    <footer className="bg-white py-6 text-sm text-gray-500">
      <div className="pointer-events-none mx-auto mb-4 h-px w-full max-w-3xl bg-gradient-to-r from-transparent via-black/20 to-transparent" />

      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 text-center md:text-left">
        <p>
          © {year} PH5. Multimodal phishing detector for text, URLs, email, and images.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4 md:justify-between">
          <div className="flex items-center gap-4">
            <a href="#privacy" className="hover:text-gray-800">
              Privacy
            </a>
            <a href="#terms" className="hover:text-gray-800">
              Terms
            </a>
          </div>
          <div className="flex items-center gap-3 text-gray-600">
            <a
              href="https://github.com/XynaxDev"
              target="_blank"
              rel="noreferrer"
              className="hover:text-gray-900"
            >
              <Github className="h-4 w-4" />
            </a>
            <a
              href="https://instagram.com/xynaxhere"
              target="_blank"
              rel="noreferrer"
              className="hover:text-gray-900"
            >
              <Instagram className="h-4 w-4" />
            </a>
            <a
              href="https://linkedin.com/in/akass7"
              target="_blank"
              rel="noreferrer"
              className="hover:text-gray-900"
            >
              <Linkedin className="h-4 w-4" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
