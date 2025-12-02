import { motion } from "framer-motion";
import { Code, Smartphone, Mail, Link2, Image, Sparkles } from "lucide-react";
import { cn } from "../lib/utils";

const modes = [
  { id: "text", label: "Text", icon: Code },
  { id: "sms", label: "SMS", icon: Smartphone },
  { id: "emails", label: "Emails", icon: Mail },
  { id: "urls", label: "URLs", icon: Link2 },
  { id: "images", label: "Images", icon: Image },
  { id: "script", label: "Script", icon: Code },
  { id: "chat", label: "Chat", icon: Sparkles },
];

export default function ModeTabs({ active, setActive }) {
  return (
    <div className="inline-flex items-center justify-center rounded-[10px] bg-black/5 px-1.5 py-1 shadow-inner">
      {modes.map((m) => {
        const Icon = m.icon;
        const isActive = active === m.id;

        return (
          <motion.button
            key={m.id}
            whileTap={{ scale: 0.96 }}
            onClick={() => setActive(m.id)}
            className={cn(
              "group flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all",
              isActive
                ? "bg-white text-black shadow-[0_0_0_1px_rgba(0,0,0,0.06),0_6px_12px_rgba(0,0,0,0.04)]"
                : "text-gray-600 hover:text-black"
            )}
          >
            <Icon
              className={cn(
                "h-3.5 w-3.5",
                isActive ? "text-orange-500" : "text-gray-400 group-hover:text-black"
              )}
            />
            <span>{m.label}</span>
          </motion.button>
        );
      })}
    </div>
  );
}
