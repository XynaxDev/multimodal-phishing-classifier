import { ArrowUpRight } from "lucide-react";

export default function AnnouncementBanner() {
  return (
    <div className="mx-auto mt-6 inline-flex items-center gap-2 rounded-full border border-gray-200 bg-gray-50 px-2 sm:px-3 py-1.5 text-xs text-gray-700 max-w-full">
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-orange-500 text-[10px] font-bold text-white flex-shrink-0">
        ✓
      </span>
      <span className="truncate">
        <strong>Latest:</strong> ph5 v2 now detects phishing in chat threads
      </span>
      <ArrowUpRight className="h-3 w-3 text-gray-500 flex-shrink-0" />
    </div>
  );
}
