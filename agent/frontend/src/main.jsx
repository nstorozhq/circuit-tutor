import React, { createContext, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot } from 'react-dom/client';
import {
  Alert, Box, Button, Collapse, CssBaseline, Divider, FormControl, FormControlLabel,
  Paper, Radio, RadioGroup, Stack, ThemeProvider, ToggleButton, Typography, createTheme,
} from '@mui/material';
import { ChatBox } from '@mui/x-chat';
import { useChat } from '@mui/x-chat/headless';
import './styles.css';

const theme = createTheme({
  palette: {
    primary: { main: '#175a4b', contrastText: '#fff' },
    background: { default: '#f6f7f6', paper: '#fff' },
    text: { primary: '#162721', secondary: '#50645a' },
    divider: '#dce4dc',
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    button: { textTransform: 'none' },
  },
});
const sessionKey = 'physics-circuit-session-v1';
const ControlsContext = createContext(null);
const features = {
  attachments: false, helperText: false, conversationList: false, conversationHeader: true,
};
const localeText = {
  composerInputPlaceholder: 'Ответ или вопрос',
  composerInputAriaLabel: 'Ответ ученика',
  composerSendButtonLabel: 'Отправить ответ',
  messageListLabel: 'Сообщения урока',
};
const authors = { assistant: 'Ассистент', user: 'Ты' };

async function api(path, init) {
  const response = await fetch(path, init);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось связаться с сервером.');
  }
  return response.json();
}

function messageForChat(message, conversationId) {
  return {
    id: message.id, conversationId, role: message.role, status: 'sent',
    parts: [{ type: 'text', text: message.text }],
  };
}

function Circuit({ circuit }) {
  if (!circuit) return null;
  return (
    <svg className="circuit" viewBox="0 0 360 198" role="img" aria-label="Учебная схема электрической цепи">
      <title>Электрическая схема с подписанными узлами и элементами</title>
      <desc>
        {circuit.junctions.map(({name, x, y}) => `Узел ${name} в точке ${x}, ${y}. `)}
        {circuit.boxes.map(({name, x, y, hidden}) => `Элемент ${name} в точке ${x}, ${y}.${hidden ? ' Правый вывод скрыт.' : ''} `)}
        {circuit.wires.map(([x1, y1, x2, y2]) => `Провод из ${x1}, ${y1} в ${x2}, ${y2}. `)}
      </desc>
      {circuit.wires.map(([x1, y1, x2, y2], index) => (
        <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} className="wire" />
      ))}
      {circuit.boxes.map(({ name, x, y, hidden }) => (
        <g key={name}>
          <rect x={x - 21} y={y - 11} width="42" height="22" rx="3" className="component" />
          <text x={x} y={y + 5} textAnchor="middle" className="component-label">{name}</text>
          {hidden && <text x={x + 35} y={y + 5} className="unknown">··· ?</text>}
        </g>
      ))}
      {circuit.junctions.map(({ name, x, y }) => (
        <g key={name}>
          <circle cx={x} cy={y} r="4" className="junction" />
          <text x={x} y={y - 12} textAnchor="middle" className="node-label">{name}</text>
        </g>
      ))}
    </svg>
  );
}

function AnswerForm({ choices, questionId, independent, busy, send }) {
  const [answer, setAnswer] = useState('');
  const [revealed, setRevealed] = useState(false);
  const optionsId = useId();
  const visible = !independent || revealed;

  function submit(event) {
    event.preventDefault();
    if (busy || !answer || !visible) return;
    send(answer);
    setAnswer('');
  }

  return (
    <>
      {independent && (
        <Button type="button" size="small" aria-expanded={visible} aria-controls={optionsId}
          disabled={busy} onClick={() => setRevealed(!revealed)}>
          {visible ? 'Скрыть варианты' : 'Показать варианты'}
        </Button>
      )}
      <Collapse in={visible} id={optionsId}>
        <Box component="form" onSubmit={submit} sx={{ pt: 0.5 }}>
          <FormControl component="fieldset" disabled={busy} fullWidth>
            <RadioGroup aria-labelledby={questionId} value={answer} name={optionsId}
              onChange={(_, value) => setAnswer(value)}>
              {choices.map((choice) => (
                <FormControlLabel key={choice} value={choice} control={<Radio size="small" />}
                  label={choice} sx={{ ml: 0, mr: 0, alignItems: 'center', minHeight: 40 }}
                  slotProps={{ typography: { variant: 'body2' } }} />
              ))}
            </RadioGroup>
          </FormControl>
          <Button type="submit" variant="contained" disableElevation size="small"
            disabled={busy || !answer} sx={{ mt: 1.5 }}>
            Ответить
          </Button>
        </Box>
      </Collapse>
    </>
  );
}

function ChatActions() {
  const { sendMessage, isStreaming } = useChat();
  const { target, state, questionId } = useContext(ControlsContext);
  if (!target) return null;
  const send = (text) => void sendMessage({ parts: [{ type: 'text', text }] }).catch(() => {});
  const inTest = state.phase === 'test';
  return createPortal(
    <>
      {state.pending && (
        <Stack direction="row" spacing={1}>
          <Button variant="contained" disableElevation size="small" disabled={isStreaming}
            onClick={() => send('Подтвердить')}>Подтвердить</Button>
          <Button size="small" disabled={isStreaming} onClick={() => send('Исправить')}>Исправить</Button>
        </Stack>
      )}
      {!!state.choices.length && (
        <AnswerForm key={`${state.phase}-${state.case_id}-${state.step_label}-${state.question}`}
          choices={state.choices} questionId={questionId} independent={state.phase === 'solo'}
          busy={isStreaming} send={send} />
      )}
      {state.can_reveal_options && (
        <Button size="small" disabled={isStreaming} onClick={() => send('Показать варианты')}>
          Показать варианты
        </Button>
      )}
      <Divider sx={{ my: 2 }} />
      <Stack direction="row" flexWrap="wrap" useFlexGap spacing={0.5}>
        {!inTest && state.phase !== 'done' && (
          <Button size="small" disabled={isStreaming} onClick={() => send('Подсказка')}>Подсказка</Button>
        )}
        {!inTest && state.case_id && <>
          <Button size="small" disabled={isStreaming} onClick={() => send('Покажи ответ')}>Эталон</Button>
          <Button size="small" disabled={isStreaming} onClick={() => send('Полный разбор')}>Полный разбор</Button>
        </>}
        {state.phase === 'done' && (
          <Button size="small" disabled={isStreaming} onClick={() => send('Резерв')}>Резервная схема</Button>
        )}
        <ToggleButton value="teacher" size="small" color="primary" selected={state.teacher} disabled={isStreaming}
          onClick={() => send(state.teacher ? 'Краткий режим' : 'Режим учителя')}>
          Режим учителя
        </ToggleButton>
      </Stack>
    </>,
    target,
  );
}

function Lesson({ session, startOver }) {
  const [state, setState] = useState(session.state);
  const [target, setTarget] = useState(null);
  const [error, setError] = useState('');
  const questionId = useId();
  const version = useRef(session.version);
  const initialMessages = useMemo(() => session.messages.map((message) => messageForChat(message, session.id)), [session]);
  const conversations = useMemo(() => [{ id: session.id, title: 'Чат' }], [session.id]);
  const adapter = useMemo(() => ({
    async sendMessage({ message, signal }) {
      setError('');
      const text = message.parts.filter((part) => part.type === 'text').map((part) => part.text).join(' ');
      try {
        const result = await api(`/api/sessions/${session.id}/turns`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message_id: message.id, version: version.current, text }),
          signal,
        });
        version.current = result.version;
        setState(result.state);
        return new ReadableStream({
          start(controller) {
            const messageId = result.message.id;
            controller.enqueue({ type: 'start', messageId });
            controller.enqueue({ type: 'text-start', id: 'reply' });
            controller.enqueue({ type: 'text-delta', id: 'reply', delta: result.message.text });
            controller.enqueue({ type: 'text-end', id: 'reply' });
            controller.enqueue({ type: 'finish', messageId });
            controller.close();
          },
        });
      } catch (failure) {
        if (failure.name !== 'AbortError') setError(failure.message);
        throw failure;
      }
    },
  }), [session.id]);

  return (
    <ControlsContext.Provider value={{ target, state, questionId }}>
      <header className="topbar">
        <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>Соединение проводников</Typography>
        <Button variant="outlined" size="small" onClick={startOver}>Новый урок</Button>
      </header>
      <main className="workspace">
        <div className="lesson-grid">
          <aside className="lesson-aside">
            <Paper component="section" variant="outlined" className="scheme-panel" aria-label="Текущая схема и этап">
              {state.position && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {state.phase === 'test' ? 'Контрольная · ' : ''}Схема {state.position} из {state.total}
                </Typography>
              )}
              {state.circuit && <div className="drawing"><Circuit circuit={state.circuit} /></div>}
              <Typography id={questionId} component="h2" variant="subtitle1" aria-live="polite"
                sx={{ fontWeight: 600, lineHeight: 1.4, mb: 1 }}>
                {state.scores ? 'Результаты' : state.question}
              </Typography>
              {state.scores && (
                <ul className="score-list">
                  {Object.entries(state.scores).map(([id, score]) => <li key={id}>{id} <strong>{score} / 4</strong></li>)}
                </ul>
              )}
              <div ref={setTarget} className="controls" />
            </Paper>
            {!session.gigachat_configured && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                GigaChat не подключён. Выбирай готовые ответы.
              </Typography>
            )}
          </aside>
          <Paper component="section" variant="outlined" className="chat-panel" aria-label="Учебный диалог">
            {error && <Alert severity="error">{error}</Alert>}
            <ChatBox
              adapter={adapter}
              initialConversations={conversations}
              initialActiveConversationId={session.id}
              initialMessages={initialMessages}
              roleDisplayNames={authors}
              localeText={localeText}
              features={features}
              slotProps={{ input: { maxLength: 1200 } }}
              slots={{ conversationHeaderActions: ChatActions }}
              sx={{ height: '100%', minHeight: 0, border: 0 }}
            />
          </Paper>
        </div>
      </main>
    </ControlsContext.Provider>
  );
}

function App() {
  const [session, setSession] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  async function start(reuse) {
    setLoading(true);
    setError('');
    try {
      const saved = reuse ? sessionStorage.getItem(sessionKey) : null;
      let current = null;
      if (saved) {
        const response = await fetch(`/api/sessions/${encodeURIComponent(saved)}`);
        if (response.ok) current = await response.json();
        else if (response.status !== 404) throw new Error('Не удалось восстановить урок.');
      }
      if (!current) current = await api('/api/sessions', { method: 'POST' });
      sessionStorage.setItem(sessionKey, current.id);
      setSession(current);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void start(true); }, []);
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {loading ? <p className="screen-note" role="status">Открываем урок…</p> :
        error ? <Alert className="screen-note" severity="error"
          action={<Button onClick={() => start(true)}>Повторить</Button>}>{error}</Alert> :
        <Lesson key={session.id} session={session} startOver={() => start(false)} />}
    </ThemeProvider>
  );
}

createRoot(document.getElementById('root')).render(<App />);
