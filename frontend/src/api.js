const API_BASE = '/api';

export async function queryAPI(query, sessionId) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'Query failed');
  return res.json();
}

export async function uploadPDF(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
  return res.json();
}

export async function getStatus() {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

export async function getSessions() {
  const res = await fetch(`${API_BASE}/sessions`);
  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  return res.json();
}

export async function submitHITL(sessionId, messageId, humanResponse) {
  const res = await fetch(`${API_BASE}/hitl/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message_id: messageId, human_response: humanResponse }),
  });
  return res.json();
}
