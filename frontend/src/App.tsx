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
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  checkApi,
  createConversation,
  getConversation,
  listConversations,
  removeConversation as removeConversationRequest,
  sendChatMessage,
  ApiRequestError,
  type ChatTurn,
  type ConversationSummary,
  type RetrievalTrace,
  type StoredMessage,
} from "./lib/api";

type Message = ChatTurn & {
  id: number | string;
  created_at: string;
  retrieval?: RetrievalTrace | null;
};
const suggestions = [
  "What are the current approaches to retrieval-augmented generation?",
  "Find papers connecting graph neural networks and scientific discovery.",
  "Compare the methods used for long-context language models.",
];

function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [graphAvailable, setGraphAvailable] = useState<boolean | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const activeConversation = conversations.find(({ id }) => id === activeId);

  useEffect(() => {
    let mounted = true;
    checkApi().then((health) => {
      if (!mounted) return;
      setApiOnline(health.apiAvailable);
      setGraphAvailable(health.graphAvailable);
      if (health.apiAvailable && !health.graphAvailable) {
        setError(
          "The knowledge graph is unavailable. New queries will be rejected until Neo4j is reachable.",
        );
      }
    });
    listConversations()
      .then(async (savedConversations) => {
        if (!mounted) return;
        setConversations(savedConversations);
        if (savedConversations.length > 0) {
          const latest = await getConversation(savedConversations[0].id);
          if (mounted) {
            setActiveId(latest.id);
            setMessages(latest.messages);
          }
        }
      })
      .catch((caught: unknown) => {
        if (mounted) {
          setApiOnline(false);
          setError(
            caught instanceof Error
              ? caught.message
              : "Could not load saved conversations.",
          );
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    function handleNewConversationShortcut(event: globalThis.KeyboardEvent) {
      if (
        !busy &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "k"
      ) {
        event.preventDefault();
        setActiveId(null);
        setMessages([]);
        setDraft("");
        setError(null);
        setMobileNavOpen(false);
        textareaRef.current?.focus();
      }
    }

    window.addEventListener("keydown", handleNewConversationShortcut);
    return () =>
      window.removeEventListener("keydown", handleNewConversationShortcut);
  }, [busy]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, busy]);

  function startConversation() {
    if (busy) return;
    setActiveId(null);
    setMessages([]);
    setDraft("");
    setError(null);
    setMobileNavOpen(false);
    textareaRef.current?.focus();
  }

  async function openConversation(id: string) {
    if (busy) return;
    setBusy(true);
    setError(null);
    setMobileNavOpen(false);
    try {
      const conversation = await getConversation(id);
      setActiveId(conversation.id);
      setMessages(conversation.messages);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not load this conversation.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function removeConversation(event: MouseEvent, id: string) {
    event.stopPropagation();
    if (busy) return;
    try {
      await removeConversationRequest(id);
      setConversations((current) => current.filter((item) => item.id !== id));
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not delete this conversation.",
      );
    }
  }

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || busy) return;

    setDraft("");
    setError(null);
    setBusy(true);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const pendingMessage = {
      id: `pending-${crypto.randomUUID()}`,
      role: "user" as const,
      content,
      created_at: new Date().toISOString(),
    };

    try {
      let conversationId = activeId;
      if (!conversationId) {
        const conversation = await createConversation();
        conversationId = conversation.id;
        setActiveId(conversation.id);
        setConversations((current) => [conversation, ...current]);
      }
      setMessages((current) => [...current, pendingMessage]);
      const response = await sendChatMessage(content, conversationId);
      setConversations((current) => [
        response.conversation,
        ...current.filter(({ id }) => id !== response.conversation.id),
      ]);
      setMessages((current) => [
        ...current.filter(({ id }) => id !== pendingMessage.id),
        response.user_message,
        response.assistant_message,
      ]);
      setApiOnline(true);
      setGraphAvailable(true);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Something went wrong while contacting ScholarGraph.",
      );
      if (caught instanceof ApiRequestError) {
        setApiOnline(true);
        if (
          caught.status === 503 &&
          caught.message.toLowerCase().includes("knowledge graph")
        ) {
          setGraphAvailable(false);
        }
      } else {
        setApiOnline(false);
      }
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
            <Atom size={23} strokeWidth={1.8} />
          </div>
          <div>
            <div className="brand-name">ScholarGraph</div>
          </div>
          <button
            className="icon-button mobile-close"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <button
          className="new-chat-button"
          onClick={startConversation}
          disabled={busy}
        >
          <Plus size={20} />
          <span>New conversation</span>
          <span className="shortcut">Ctrl K</span>
        </button>

        <div className="sidebar-section-label">
          <span>CONVERSATION HISTORY</span>
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
                  disabled={busy}
                  onClick={() => void openConversation(conversation.id)}
                >
                  <MessageSquareText size={16} />
                  <span className="conversation-title">
                    {conversation.title}
                  </span>
                </button>
                <button
                  className="conversation-delete"
                  aria-label={`Delete ${conversation.title}`}
                  disabled={busy}
                  onClick={(event) =>
                    void removeConversation(event, conversation.id)
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
              <strong>Knowledge graph</strong>
              <span>
                {apiOnline === null
                  ? "Checking connection"
                  : !apiOnline
                    ? "Backend unavailable"
                    : graphAvailable
                      ? "Neo4j available"
                      : "Neo4j unavailable · responses paused"}
              </span>
            </div>
            <span
              className={`status-dot ${apiOnline && graphAvailable ? "online" : apiOnline === false || graphAvailable === false ? "offline" : ""}`}
            />
          </div>
          <div className="local-note">
            <HardDrive size={15} />
            <span>Chat history saved locally.</span>
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
              ScholarGraph Chatbot
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
                    <div className="message-content">
                      {message.role === "assistant" ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      ) : (
                        message.content
                      )}
                    </div>
                    {message.role === "assistant" && message.retrieval && (
                      <details className="retrieval-evidence">
                        <summary>
                          <span>Retrieval evidence</span>
                          <span className="retrieval-counts">
                            Neo4j {message.retrieval.graph_result_count} rows
                            <span aria-hidden="true"> · </span>
                            ChromaDB {message.retrieval.vector_hit_count} papers
                          </span>
                        </summary>
                        <div className="retrieval-body">
                          <section className="retrieval-section">
                            <h4>Knowledge graph</h4>
                            {message.retrieval.graph_queries.map(
                              (attempt, index) => (
                                <details
                                  className="retrieval-query"
                                  key={`${attempt.kind}-${index}`}
                                >
                                  <summary>
                                    {attempt.kind} · {attempt.result_count} rows
                                  </summary>
                                  <pre>
                                    <code>{attempt.cypher}</code>
                                  </pre>
                                  {attempt.error && (
                                    <p className="retrieval-note">
                                      {attempt.error}
                                    </p>
                                  )}
                                  {attempt.parameters?.terms && (
                                    <p className="retrieval-note">
                                      Search terms:{" "}
                                      {String(attempt.parameters.terms)}
                                    </p>
                                  )}
                                </details>
                              ),
                            )}
                            {message.retrieval.graph_results_truncated && (
                              <p className="retrieval-note">
                                Showing the first 20 of{" "}
                                {message.retrieval.graph_result_count} graph
                                rows.
                              </p>
                            )}
                            {message.retrieval.graph_records.length === 0 ? (
                              <p className="retrieval-note">
                                The graph traversal ran but returned no matching
                                rows.
                              </p>
                            ) : (
                              <div className="graph-records">
                                {message.retrieval.graph_records.map(
                                  (record, index) => (
                                    <details
                                      className="graph-record"
                                      key={index}
                                    >
                                      <summary>Graph row {index + 1}</summary>
                                      <pre>
                                        <code>
                                          {JSON.stringify(record, null, 2)}
                                        </code>
                                      </pre>
                                    </details>
                                  ),
                                )}
                              </div>
                            )}
                          </section>
                          <section className="retrieval-section">
                            <h4>Retrieved papers</h4>
                            {message.retrieval.vector_sources.map((source) => (
                              <article
                                className="retrieval-source"
                                key={source.paper_id}
                              >
                                <strong>
                                  {source.title ?? "Untitled paper"}
                                </strong>
                                <span>
                                  [{source.paper_id}]
                                  {source.published
                                    ? ` · ${source.published}`
                                    : ""}
                                </span>
                                <p>{source.excerpt}</p>
                              </article>
                            ))}
                          </section>
                        </div>
                      </details>
                    )}
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
