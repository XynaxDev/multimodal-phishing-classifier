import { motion } from "framer-motion";

export default function PremiumHero() {
  return (
    <section className="mx-auto mt-10 w-full max-w-3xl px-4 text-center">
      <motion.h1
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="text-4xl font-black tracking-tight text-black sm:text-5xl md:text-6xl"
      >
        Turn messages into {" "}
        <span className="text-orange-500">LLM‑ready verdicts</span>.
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.12, duration: 0.25 }}
        className="mt-3 text-sm text-gray-600 md:text-base"
      >
        Paste suspicious text, links, or emails — or upload screenshots and
        chats — and PH5 tells you how phishy they look.
      </motion.p>
    </section>
  );
}
