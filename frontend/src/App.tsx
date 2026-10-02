import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import {
  ArrowUp,
  Atom,
  BookOpen,
  Check,
  ChevronDown,
  Database,
  HardDrive,
  Menu,
  MessageSquareText,
  Plus,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import type { MouseEvent } from "react";
import { checkApi, sendChatMessage, type ChatTurn } from "./lib/api";

type Message = ChatTurn & {
  id: string;
  createdAt: number;
};

type Conversation = {
  id: string;
  title: string;
  updatedAt: number;
  messages: Message[];
};

const storageKey = "scholargraph.conversations.v1";
const suggestions = [
  "What are the current approaches to retrieval-augmented generation?",
  "Find papers connecting graph neural networks and scientific discovery.",
  "Compare the methods used for long-context language models.",
];

function readConversations(): Conversation[] {
  try {
    const saved = localStorage.getItem(storageKey);
    return saved ? (JSON.parse(saved) as Conversation[]) : [];
  } catch {
    return [];
  }
}

function makeMessage(role: Message["role"], content: string): Message {
  return { id: crypto.randomUUID(), role, content, createdAt: Date.now() };
}

function App() {
  const [conversations, setConversations] = useState(readConversations);
  const [activeId, setActiveId] = useState<string | null>(
    () => readConversations()[0]?.id ?? null,
  );
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeConversation = conversations.find(({ id }) => id === activeId);
  const messages = activeConversation?.messages ?? [];

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    let mounted = true;
    checkApi().then((online) => {
      if (mounted) setApiOnline(online);
    });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    function handleNewConversationShortcut(event: globalThis.KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setActiveId(null);
        setDraft("");
        setError(null);
        setMobileNavOpen(false);
        textareaRef.current?.focus();
      }
    }

    window.addEventListener("keydown", handleNewConversationShortcut);
    return () =>
      window.removeEventListener("keydown", handleNewConversationShortcut);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, busy]);

  function startConversation() {
    setActiveId(null);
    setDraft("");
    setError(null);
    setMobileNavOpen(false);
    textareaRef.current?.focus();
  }

  function removeConversation(event: MouseEvent, id: string) {
    event.stopPropagation();
    setConversations((current) => current.filter((item) => item.id !== id));
    if (activeId === id) setActiveId(null);
  }

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || busy) return;

    const currentId = activeId ?? crypto.randomUUID();
    const userMessage = makeMessage("user", content);
    const currentConversation = conversations.find(
      ({ id }) => id === currentId,
    );
    const nextMessages = [
      ...(currentConversation?.messages ?? []),
      userMessage,
    ];
    const conversation: Conversation = {
      id: currentId,
      title: currentConversation?.title ?? content.slice(0, 48),
      updatedAt: Date.now(),
      messages: nextMessages,
    };

    setActiveId(currentId);
    setConversations((current) => [
      conversation,
      ...current.filter(({ id }) => id !== currentId),
    ]);
    setDraft("");
    setError(null);
    setBusy(true);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      const history = (currentConversation?.messages ?? [])
        .slice(-12)
        .map(({ role, content: turnContent }) => ({
          role,
          content: turnContent,
        }));
      const answer = await sendChatMessage(content, history);
      const assistantMessage = makeMessage("assistant", answer);
      setConversations((current) =>
        current.map((item) =>
          item.id === currentId
            ? {
                ...item,
                updatedAt: Date.now(),
                messages: [...item.messages, assistantMessage],
              }
            : item,
        ),
      );
      setApiOnline(true);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Something went wrong while contacting ScholarGraph.",
      );
      setApiOnline(false);
    } finally {
      setBusy(false);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage();
    }
  }

  function handleDraftChange(value: string) {
    setDraft(value);
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    }
  }

  function renderSidebar() {
    return (
      <>
        <div className="brand-lockup">
          <div className="brand-mark">
            <Atom size={21} strokeWidth={1.8} />
          </div>
          <div>
            <div className="brand-name">ScholarGraph</div>
            <div className="brand-caption">RESEARCH STUDIO</div>
          </div>
          <button
            className="icon-button mobile-close"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <button className="new-chat-button" onClick={startConversation}>
          <Plus size={17} />
          <span>New conversation</span>
          <span className="shortcut">Ctrl K</span>
        </button>

        <div className="sidebar-section-label">
          <span>YOUR WORKSPACE</span>
          <ChevronDown size={13} />
        </div>
        <div className="conversation-list">
          {conversations.length === 0 ? (
            <p className="empty-history">
              Your research conversations will appear here.
            </p>
          ) : (
            conversations.map((conversation) => (
              <div
                className={`conversation-item ${activeId === conversation.id ? "active" : ""}`}
                key={conversation.id}
              >
                <button
                  className="conversation-select"
                  onClick={() => {
                    setActiveId(conversation.id);
                    setError(null);
                    setMobileNavOpen(false);
                  }}
                >
                  <MessageSquareText size={16} />
                  <span className="conversation-title">
                    {conversation.title}
                  </span>
                </button>
                <button
                  className="conversation-delete"
                  aria-label={`Delete ${conversation.title}`}
                  onClick={(event) =>
                    removeConversation(event, conversation.id)
                  }
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="connection-card">
            <div className="connection-icon">
              <Database size={16} />
            </div>
            <div className="connection-copy">
              <strong>Backend API</strong>
              <span>
                {apiOnline === null
                  ? "Checking connection"
                  : apiOnline
                    ? "Ready to receive queries"
                    : "Offline · start the API"}
              </span>
            </div>
            <span
              className={`status-dot ${apiOnline ? "online" : apiOnline === false ? "offline" : ""}`}
            />
          </div>
          <div className="local-note">
            <HardDrive size={15} />
            <span>Chat history stays on this device</span>
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="app-shell">
      {mobileNavOpen && (
        <button
          className="mobile-scrim"
          onClick={() => setMobileNavOpen(false)}
          aria-label="Close navigation"
        />
      )}
      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        {renderSidebar()}
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-leading">
            <button
              className="icon-button mobile-menu"
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={18} />
            </button>
            <div className="breadcrumb">
              <span>Workspace</span>
              <span className="breadcrumb-separator">/</span>
              <strong>{activeConversation?.title ?? "New conversation"}</strong>
            </div>
          </div>
          <div className="topbar-trailing">
            <span className="model-indicator">
              <span className="model-orb">
                <Sparkles size={13} />
              </span>{" "}
              ScholarGraph AI
            </span>
            <span className="topbar-divider" />
            <div className="avatar" title="Research workspace">
              S
            </div>
          </div>
        </header>

        <div
          className={`chat-scroll ${messages.length === 0 ? "chat-scroll-empty" : ""}`}
        >
          {messages.length === 0 ? (
            <section className="welcome-view">
              <div className="welcome-eyebrow">
                <span className="eyebrow-line" /> YOUR RESEARCH, CONNECTED
              </div>
              <h1>
                Where should we
                <br />
                <em>begin?</em>
              </h1>
              <p className="welcome-copy">
                Ask a question across papers, methods, and research connections.
                ScholarGraph brings your graph and literature together.
              </p>

              <div className="suggestion-heading">
                <span>START WITH A QUESTION</span>
                <BookOpen size={15} />
              </div>
              <div className="suggestion-list">
                {suggestions.map((suggestion, index) => (
                  <button
                    className="suggestion-card"
                    key={suggestion}
                    onClick={() => {
                      setDraft(suggestion);
                      textareaRef.current?.focus();
                    }}
                  >
                    <span className={`suggestion-index index-${index + 1}`}>
                      0{index + 1}
                    </span>
                    <span>{suggestion}</span>
                    <ArrowUp className="suggestion-arrow" size={15} />
                  </button>
                ))}
              </div>
              <div className="capability-note">
                <Check size={14} />
                <span>Answers grounded in your indexed research library</span>
              </div>
            </section>
          ) : (
            <section className="message-thread" aria-live="polite">
              <div className="thread-intro">
                <span className="thread-rule" />
                <span>RESEARCH CONVERSATION</span>
                <span className="thread-rule" />
              </div>
              {messages.map((message) => (
                <article
                  className={`message-row ${message.role}`}
                  key={message.id}
                >
                  {message.role === "assistant" && (
                    <div className="assistant-avatar">
                      <Atom size={17} />
                    </div>
                  )}
                  <div className="message-content-wrap">
                    <div className="message-label">
                      {message.role === "assistant" ? "SCHOLARGRAPH" : "YOU"}
                    </div>
                    <div className="message-content">{message.content}</div>
                  </div>
                  {message.role === "user" && (
                    <div className="user-avatar">Y</div>
                  )}
                </article>
              ))}
              {busy && (
                <article className="message-row assistant">
                  <div className="assistant-avatar">
                    <Atom size={17} />
                  </div>
                  <div className="message-content-wrap">
                    <div className="message-label">
                      SCHOLARGRAPH{" "}
                      <span className="thinking-label">
                        SEARCHING YOUR LIBRARY
                      </span>
                    </div>
                    <div className="typing-indicator">
                      <i />
                      <i />
                      <i />
                    </div>
                  </div>
                </article>
              )}
              {error && (
                <div className="error-banner" role="alert">
                  <span>{error}</span>
                  <button
                    onClick={() => setError(null)}
                    aria-label="Dismiss error"
                  >
                    <X size={15} />
                  </button>
                </div>
              )}
              <div ref={bottomRef} />
            </section>
          )}
        </div>

        <footer className="composer-area">
          {error && messages.length === 0 && (
            <div className="error-banner composer-error" role="alert">
              <span>{error}</span>
              <button onClick={() => setError(null)} aria-label="Dismiss error">
                <X size={15} />
              </button>
            </div>
          )}
          <form className="composer" onSubmit={submitMessage}>
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(event) => handleDraftChange(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask ScholarGraph anything about your research..."
              rows={1}
              aria-label="Message ScholarGraph"
              maxLength={4000}
              disabled={busy}
            />
            <div className="composer-controls">
              <div className="composer-hint">
                <span className="keycap">↵</span> to send{" "}
                <span className="hint-separator">·</span>{" "}
                <span className="keycap">⇧ ↵</span> new line
              </div>
              <div className="composer-actions">
                <span className="character-count">
                  {draft.length > 0 ? `${draft.length}/4000` : ""}
                </span>
                <button
                  className="send-button"
                  type="submit"
                  disabled={!draft.trim() || busy}
                  aria-label="Send message"
                >
                  <ArrowUp size={18} />
                </button>
              </div>
            </div>
          </form>
          <div className="disclaimer">
            ScholarGraph can make mistakes. Verify important findings against
            the original papers.
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
