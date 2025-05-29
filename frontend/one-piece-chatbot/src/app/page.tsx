"use client";
import React, { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Use the original, confirmed avatar paths
const AVATARS = {
  user: "/avatar-user.png",
  bot: "/Straw-Hat-Logo.png",
};

export default function OnePieceChatbot() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, loading]);

  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!input.trim()) return;

  const newMessage = { role: "user", content: input };
  setMessages((prev) => [...prev, newMessage]);
  setInput("");
  setLoading(true);

  try {
    const res = await fetch("https://apc9mq1dacoul2-8000.proxy.runpod.net/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: input,
        history: []
      }),
    });

    const data = await res.json();
    setMessages((prev) => [...prev, { role: "bot", content: data.answer }]);
  } catch (err) {
    setMessages((prev) => [
      ...prev,
      { role: "bot", content: "⚠️ Error: Unable to connect. Please try again." },
    ]);
  } finally {
    setLoading(false);
  }
};


  const quickReplies = [
    "Who leads the Straw Hat Pirates?",
    "Describe Nami's role in the crew.",
    "What is the One Piece treasure?",
    "List the current Straw Hat crew members.",
  ];

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-8 px-4 font-sans">
      {/* Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="fixed top-0 left-0 w-full bg-white shadow-md z-20"
      >
        <div className="max-w-5xl mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <img
              src="/Straw-Hat-Logo.png"
              alt="Straw Hat Logo"
              className="h-10 w-10 object-contain"
            />
            <h1 className="text-2xl font-semibold text-gray-900">
              One Piece Assistant
            </h1>
          </div>
          <nav className="text-sm text-gray-600">
            <a
              href="https://onepiece.fandom.com/wiki/One_Piece"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-blue-600 transition"
              aria-label="Visit One Piece Wiki"
            >
              Explore Wiki
            </a>
          </nav>
        </div>
      </motion.header>

      {/* Chat Container */}
      <main
        ref={chatRef}
        className="flex-1 w-full max-w-5xl mx-auto mt-20 mb-24 bg-white rounded-xl shadow-lg overflow-y-auto px-6 py-8"
        style={{ maxHeight: "calc(100vh - 160px)" }}
        role="log"
        aria-live="polite"
      >
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col items-center justify-center h-full text-center"
          >
            <img
              src="/Straw-Hat-Logo.png"
              alt="Straw Hat Logo"
              className="h-16 w-16 mb-4"
            />
            <h2 className="text-xl font-medium text-gray-900">
              One Piece Assistant
            </h2>
            <p className="text-gray-600 mt-2 max-w-md">
              Dive into the world of One Piece. Ask about the Straw Hat Pirates or the Grand Line.
            </p>
          </motion.div>
        )}

        <AnimatePresence>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex items-start gap-4 mb-4 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "bot" && (
                <img
                  src={AVATARS.bot}
                  alt="Assistant Avatar"
                  className="h-10 w-10 rounded-full border border-gray-200"
                  aria-hidden="true"
                />
              )}
              <div
                className={`px-5 py-3 rounded-lg text-base max-w-[70%] shadow-sm ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white rounded-br-none"
                    : "bg-gray-50 text-gray-900 rounded-bl-none border border-gray-200"
                }`}
              >
                {msg.content}
              </div>
              {msg.role === "user" && (
                <img
                  src={AVATARS.user}
                  alt="User Avatar"
                  className="h-10 w-10 rounded-full border border-gray-200"
                  aria-hidden="true"
                />
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
            className="flex items-center gap-3"
          >
            <img
              src={AVATARS.bot}
              alt="Assistant Avatar"
              className="h-10 w-10 rounded-full border border-gray-200"
            />
            <span className="text-gray-600 font-medium">
              Processing your request...
            </span>
          </motion.div>
        )}
      </main>

      {/* Quick Replies */}
      {messages.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-5xl mx-auto flex flex-wrap gap-3 justify-center mb-6"
        >
          {quickReplies.map((reply) => (
            <button
              key={reply}
              className="bg-gray-100 text-gray-900 text-sm font-medium px-5 py-2 rounded-lg hover:bg-blue-100 hover:text-blue-600 transition duration-150"
              onClick={() => setInput(reply)}
              type="button"
              aria-label={`Quick reply: ${reply}`}
            >
              {reply}
            </button>
          ))}
        </motion.div>
      )}

      {/* Input Bar */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="fixed bottom-0 left-0 w-full bg-white border-t border-gray-200 py-4"
      >
        <form
          onSubmit={handleSubmit}
          className="max-w-5xl mx-auto px-6 flex items-center gap-3"
          autoComplete="off"
        >
          <input
            type="text"
            className="flex-grow bg-gray-50 border border-gray-200 rounded-lg px-5 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 placeholder:text-gray-400 text-base transition duration-150"
            placeholder="Ask about the One Piece universe..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            aria-label="Message input"
          />
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-3 rounded-lg transition duration-150 disabled:opacity-50 flex items-center gap-2"
            aria-label="Send message"
          >
            {loading ? (
              <svg
                className="h-5 w-5 animate-spin"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8z"
                />
              </svg>
            ) : (
              <span className="flex items-center gap-2">
                Send
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </span>
            )}
          </motion.button>
        </form>
      </motion.div>

      {/* Footer */}
      <footer className="mt-4 text-sm text-gray-500 text-center">
        Created for One Piece fans ·{" "}
        <a
          href="https://onepiece.fandom.com/wiki/Straw_Hat_Pirates"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-blue-600 transition"
          aria-label="Learn more about the Straw Hat Pirates"
        >
          Learn More
        </a>
      </footer>
    </div>
  );
}