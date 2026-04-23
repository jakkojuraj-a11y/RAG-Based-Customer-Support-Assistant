import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Plus, Menu, X, Upload, FileText, ChevronDown, ChevronUp, Copy, RotateCcw, Settings, Sparkles, Bot, User, AlertTriangle, CheckCircle, Clock, Zap, Database, Shield } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { queryAPI, uploadPDF, getStatus, getDocuments } from './api';

/* ── Typing Indicator ─────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3">
      <div className="w-2 h-2 rounded-full bg-primary typing-dot" />
      <div className="w-2 h-2 rounded-full bg-primary typing-dot" />
      <div className="w-2 h-2 rounded-full bg-primary typing-dot" />
    </div>
  );
}

/* ── Confidence Bar ───────────────────────────────────────── */
function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? 'bg-accent-green' : pct >= 50 ? 'bg-accent-amber' : 'bg-accent-red';
  return (
    <div className="flex items-center gap-2 text-xs text-text-dim">
      <span>Confidence</span>
      <div className="flex-1 h-1.5 rounded-full bg-surface-3 max-w-[120px]">
        <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 0.8 }} className={`h-full rounded-full ${color}`} />
      </div>
      <span className="font-medium text-text">{pct}%</span>
    </div>
  );
}

/* ── Source Card ───────────────────────────────────────────── */
function SourceCard({ source, index }) {
  return (
    <div className="glass rounded-lg p-3 text-xs">
      <div className="flex items-center gap-2 mb-1">
        <FileText size={12} className="text-primary-light" />
        <span className="font-medium text-text">Source {index + 1}</span>
        <span className="text-text-dim">Page {source.page}</span>
      </div>
      <p className="text-text-dim leading-relaxed">{source.excerpt}</p>
    </div>
  );
}

/* ── Chat Message ─────────────────────────────────────────── */
function ChatMessage({ msg, onCopy, onRegenerate }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = msg.role === 'user';
  const isEscalated = msg.escalated;

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''} group`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${isUser ? 'bg-primary/20' : 'bg-surface-3'}`}>
        {isUser ? <User size={16} className="text-primary-light" /> : <Bot size={16} className="text-primary-light" />}
      </div>
      {/* Bubble */}
      <div className={`max-w-[75%] ${isUser ? 'ml-auto' : ''}`}>
        <div className={`rounded-2xl px-4 py-3 ${isUser ? 'bg-primary/15 border border-primary/20' : 'glass'}`}>
          {isEscalated && (
            <div className="flex items-center gap-2 mb-2 px-2 py-1 rounded-lg bg-accent-amber/10 border border-accent-amber/20 text-accent-amber text-xs font-medium">
              <AlertTriangle size={12} /> Escalated to Human Agent
            </div>
          )}
          <div className="text-sm leading-relaxed prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        </div>
        {/* Meta row for assistant */}
        {!isUser && msg.confidence !== undefined && (
          <div className="mt-2 space-y-2">
            <div className="flex items-center gap-3 flex-wrap">
              <ConfidenceBar value={msg.confidence} />
              {msg.intent && <span className="text-xs px-2 py-0.5 rounded-full bg-surface-3 text-text-dim">{msg.intent}</span>}
              {msg.handled_by && <span className="text-xs px-2 py-0.5 rounded-full bg-surface-3 text-text-dim">{msg.handled_by === 'ai' ? '🤖 AI' : '👤 Human'}</span>}
              {msg.latency_ms && <span className="text-xs text-text-dim">{msg.latency_ms}ms</span>}
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => onCopy(msg.content)} className="text-text-dim hover:text-text transition-colors p-1 rounded hover:bg-surface-3" title="Copy"><Copy size={13} /></button>
              <button onClick={() => onRegenerate(msg.query)} className="text-text-dim hover:text-text transition-colors p-1 rounded hover:bg-surface-3" title="Regenerate"><RotateCcw size={13} /></button>
              {msg.sources?.length > 0 && (
                <button onClick={() => setShowSources(!showSources)} className="flex items-center gap-1 text-xs text-text-dim hover:text-primary-light transition-colors">
                  <Database size={12} /> {msg.sources.length} sources {showSources ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </button>
              )}
            </div>
            <AnimatePresence>
              {showSources && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="space-y-2 overflow-hidden">
                  {msg.sources.map((s, i) => <SourceCard key={i} source={s} index={i} />)}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
        <p className="text-[10px] text-text-dim/50 mt-1 px-1">{msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}</p>
      </div>
    </motion.div>
  );
}

/* ── Sidebar ──────────────────────────────────────────────── */
function Sidebar({ open, onClose, sessions, onNewChat, activeSession, onSelectSession, status, documents, onUpload }) {
  const [uploading, setUploading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const fileRef = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try { await onUpload(file); } catch (err) { alert(err.message); }
    setUploading(false);
    e.target.value = '';
  };

  return (
    <>
      {open && <div className="fixed inset-0 bg-black/50 z-30 lg:hidden" onClick={onClose} />}
      <motion.aside initial={false} animate={{ x: open ? 0 : -320 }} transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className="fixed lg:relative z-40 w-[300px] h-full flex flex-col bg-surface border-r border-border shrink-0">
        {/* Header */}
        <div className="p-4 flex items-center justify-between border-b border-border">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-primary/20 flex items-center justify-center"><Sparkles size={16} className="text-primary-light" /></div>
            <div><h1 className="text-sm font-semibold gradient-text">TechCorp AI</h1><p className="text-[10px] text-text-dim">Support Assistant</p></div>
          </div>
          <button onClick={onClose} className="lg:hidden p-1 rounded-lg hover:bg-surface-3 text-text-dim"><X size={18} /></button>
        </div>
        {/* New Chat */}
        <div className="p-3">
          <button onClick={onNewChat} className="w-full flex items-center gap-2 px-4 py-2.5 rounded-xl bg-primary/10 border border-primary/20 text-primary-light hover:bg-primary/20 transition-all text-sm font-medium">
            <Plus size={16} /> New Chat
          </button>
        </div>
        {/* Chat History */}
        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          <p className="text-[10px] uppercase tracking-wider text-text-dim px-2 py-2 font-medium">Recent Chats</p>
          {sessions.map(s => (
            <button key={s.id} onClick={() => onSelectSession(s.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-all ${s.id === activeSession ? 'bg-surface-3 text-text' : 'text-text-dim hover:bg-surface-2 hover:text-text'}`}>
              {s.title}
            </button>
          ))}
          {sessions.length === 0 && <p className="text-xs text-text-dim/50 px-2 py-4">No conversations yet</p>}
        </div>
        {/* Knowledge Base */}
        <div className="p-3 border-t border-border space-y-2">
          <p className="text-[10px] uppercase tracking-wider text-text-dim font-medium px-1">Knowledge Base</p>
          <input type="file" ref={fileRef} accept=".pdf" onChange={handleUpload} className="hidden" />
          <button onClick={() => fileRef.current?.click()} disabled={uploading}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl border border-dashed border-border text-text-dim hover:border-primary/40 hover:text-primary-light transition-all text-xs">
            <Upload size={14} /> {uploading ? 'Uploading...' : 'Upload PDF'}
          </button>
          {documents.map((d, i) => (
            <div key={i} className="flex items-center gap-2 px-2 py-1 text-xs text-text-dim">
              <FileText size={12} className="text-primary-light shrink-0" /> <span className="truncate">{d.name}</span> <span className="ml-auto shrink-0">{d.size_kb}KB</span>
            </div>
          ))}
        </div>
        {/* Status */}
        <div className="p-3 border-t border-border">
          <div className="flex items-center gap-2 text-xs text-text-dim">
            <div className={`w-2 h-2 rounded-full ${status.ready ? 'bg-accent-green animate-pulse' : 'bg-accent-red'}`} />
            <span>{status.ready ? 'Online' : 'Offline'}</span>
            <span className="ml-auto">{status.documents || 0} docs</span>
          </div>
          <p className="text-[10px] text-text-dim/50 mt-1">{status.model || 'N/A'}</p>
        </div>
      </motion.aside>
    </>
  );
}

/* ── Main App ─────────────────────────────────────────────── */
export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [status, setStatus] = useState({});
  const [documents, setDocuments] = useState([]);
  const chatRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => { chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' }); };
  useEffect(() => { scrollToBottom(); }, [messages, loading]);

  const refreshStatus = useCallback(async () => {
    try { const s = await getStatus(); setStatus(s); } catch {}
    try { const d = await getDocuments(); setDocuments(d.documents || []); } catch {}
  }, []);

  useEffect(() => { refreshStatus(); const i = setInterval(refreshStatus, 15000); return () => clearInterval(i); }, [refreshStatus]);

  const handleNewChat = () => { setMessages([]); setSessionId(null); };

  const handleSend = async (queryOverride) => {
    const q = queryOverride || input.trim();
    if (!q || loading) return;
    if (!queryOverride) setInput('');

    const userMsg = { role: 'user', content: q, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await queryAPI(q, sessionId);
      if (!sessionId) setSessionId(res.session_id);
      const assistantMsg = { role: 'assistant', content: res.response, confidence: res.confidence, intent: res.intent, handled_by: res.handled_by, escalated: res.escalated, escalation_reason: res.escalation_reason, sources: res.sources || [], latency_ms: res.latency_ms, query: q, timestamp: res.timestamp };
      setMessages(prev => [...prev, assistantMsg]);
      setSessions(prev => {
        const existing = prev.find(s => s.id === res.session_id);
        if (existing) return prev;
        return [{ id: res.session_id, title: q.slice(0, 60) }, ...prev];
      });
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}`, confidence: 0, timestamp: new Date().toISOString() }]);
    }
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleCopy = (text) => { navigator.clipboard.writeText(text); };
  const handleRegenerate = (query) => { if (query) handleSend(query); };
  const handleUpload = async (file) => { await uploadPDF(file); refreshStatus(); };
  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };

  return (
    <div className="h-screen flex overflow-hidden">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} sessions={sessions} onNewChat={handleNewChat} activeSession={sessionId} onSelectSession={() => {}} status={status} documents={documents} onUpload={handleUpload} />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        {/* Top Bar */}
        <header className="h-14 flex items-center px-4 border-b border-border shrink-0 glass">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 rounded-lg hover:bg-surface-3 text-text-dim mr-3"><Menu size={18} /></button>
          <div className="flex items-center gap-2">
            <Sparkles size={16} className="text-primary-light" />
            <h2 className="text-sm font-semibold">TechCorp Support Assistant</h2>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status.ready ? 'bg-accent-green' : 'bg-accent-red'}`} />
            <span className="text-xs text-text-dim hidden sm:inline">{status.ready ? 'Ready' : 'Offline'}</span>
          </div>
        </header>

        {/* Messages */}
        <div ref={chatRef} className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center px-4">
              <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }} className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                  <Sparkles size={28} className="text-primary-light" />
                </div>
                <h2 className="text-2xl font-bold mb-2 gradient-text">How can I help you?</h2>
                <p className="text-text-dim text-sm mb-8">Ask about TechCorp products, troubleshooting, billing, or policies.</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {['How do I set up my SmartHome Hub?', 'What is your refund policy?', 'My VPN is slow', 'I want to talk to a human'].map((q, i) => (
                    <motion.button key={i} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={() => { setInput(q); setTimeout(() => handleSend(q), 50); }}
                      className="text-left px-4 py-3 rounded-xl glass text-sm text-text-dim hover:text-text hover:border-primary/30 transition-all">
                      {q}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              {messages.map((msg, i) => <ChatMessage key={i} msg={msg} onCopy={handleCopy} onRegenerate={handleRegenerate} />)}
              {loading && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-xl bg-surface-3 flex items-center justify-center"><Bot size={16} className="text-primary-light" /></div>
                  <div className="glass rounded-2xl"><TypingIndicator /></div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-border shrink-0">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2 glass rounded-2xl p-2">
              <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="Ask anything about TechCorp..."
                rows={1} className="flex-1 bg-transparent border-none outline-none resize-none text-sm text-text placeholder:text-text-dim/50 py-2 px-3 max-h-32" />
              <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => handleSend()} disabled={!input.trim() || loading}
                className="p-2.5 rounded-xl bg-primary hover:bg-primary-dark disabled:opacity-30 disabled:cursor-not-allowed transition-all">
                <Send size={16} className="text-white" />
              </motion.button>
            </div>
            <p className="text-[10px] text-text-dim/40 text-center mt-2">RAG-powered by LangGraph + ChromaDB + Gemini</p>
          </div>
        </div>
      </main>
    </div>
  );
}
