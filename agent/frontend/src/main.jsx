import React, { createContext, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Alert, Box, Button, Collapse, CssBaseline, Dialog, DialogActions, DialogContent, DialogTitle,
  Divider, FormControl, FormControlLabel, Paper, Radio, RadioGroup, Stack, TextField, ThemeProvider,
  ToggleButton, Typography, createTheme,
} from '@mui/material';
import { ChatBox, ChatMessageContent } from '@mui/x-chat';
import { useChat, useMessageContext } from '@mui/x-chat/headless';
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
  attachments: false, helperText: false, conversationList: false, conversationHeader: false,
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
    throw Object.assign(new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось связаться с сервером.'),
      { status: response.status });
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

function AnswerForm({ choices, question, independent, busy, send }) {
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
            <RadioGroup aria-label={question} value={answer} name={optionsId}
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

function ChatActions({ state }) {
  const { sendMessage, isStreaming } = useChat();
  const send = (text) => void sendMessage({ parts: [{ type: 'text', text }] }).catch(() => {});
  const inTest = state.phase === 'test';
  return (
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
          choices={state.choices} question={state.question} independent={state.phase === 'solo'}
          busy={isStreaming} send={send} />
      )}
      {state.can_reveal_options && (
        <Button size="small" disabled={isStreaming} onClick={() => send('Показать варианты')}>
          Показать варианты
        </Button>
      )}
      <Divider sx={{ my: 1.5 }} />
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
    </>
  );
}

function LessonMessage({ afterContent, ...props }) {
  const { messageId, message } = useMessageContext();
  const { state, activeMessageId } = useContext(ControlsContext);
  const alreadyAsks = message?.parts.some(part => part.type === 'text' && part.text.includes(state.question));
  const active = messageId === activeMessageId;
  return (
    <ChatMessageContent {...props} afterContent={
      <>
        {active && (
          <Box className="lesson-controls">
            {state.position && state.circuit && (
              <Typography variant="caption" color="text.secondary">
                {state.phase === 'test' ? 'Контрольная · ' : ''}Схема {state.position} из {state.total}
              </Typography>
            )}
            {state.circuit && state.phase !== 'method' && (
              <div className="drawing"><Circuit circuit={state.circuit} /></div>
            )}
            {!alreadyAsks && !state.pending && !state.scores && (
              <Typography sx={{ mt: 0.75 }} variant="body2">{state.question}</Typography>
            )}
            <ChatActions state={state} />
          </Box>
        )}
        {afterContent}
      </>
    } />
  );
}

function Lesson({ session }) {
  const [state, setState] = useState(session.state);
  const [activeMessageId, setActiveMessageId] = useState(session.messages.at(-1).id);
  const [error, setError] = useState('');
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
        setActiveMessageId(result.message.id);
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
    <ControlsContext.Provider value={{ state, activeMessageId }}>
      <main className="workspace">
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
            slotProps={{ composerInput: { maxLength: 1200 } }}
            slots={{ messageContent: LessonMessage }}
            sx={{ height: '100%', minHeight: 0, border: 0 }}
          />
        </Paper>
      </main>
    </ControlsContext.Provider>
  );
}

function AccountDialog({ open, close, onAccount }) {
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function dismiss() {
    if (busy) return;
    setError('');
    setRegister(false);
    close();
  }

  async function submit(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    setBusy(true);
    setError('');
    try {
      const profile = await api(register ? '/api/auth/register' : '/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: data.get('username'),
          password: data.get('password'),
          guest_lesson: sessionStorage.getItem(sessionKey),
        }),
      });
      setRegister(false);
      onAccount(profile);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onClose={dismiss} fullWidth maxWidth="xs" aria-labelledby="account-title">
      <DialogTitle id="account-title">{register ? 'Создать аккаунт' : 'Вход'}</DialogTitle>
      <Box key={String(register)} component="form" onSubmit={submit}>
        <DialogContent sx={{ pt: 1 }}>
          <Stack spacing={2}>
            {error && <Alert severity="error">{error}</Alert>}
            <TextField label="Логин" name="username" autoComplete="username" autoFocus required
              helperText={register ? 'От 3 до 32 символов. Латинские буквы, цифры, точка, дефис или подчёркивание.' : undefined}
              slotProps={{ htmlInput: { pattern: '[A-Za-z0-9_.\\-]{3,32}', maxLength: 32, autoCapitalize: 'none' } }} />
            <TextField label="Пароль" name="password" type="password" required
              autoComplete={register ? 'new-password' : 'current-password'}
              helperText={register ? 'Не меньше 12 символов.' : undefined}
              slotProps={{ htmlInput: { minLength: 12, maxLength: 128 } }} />
          </Stack>
        </DialogContent>
        <DialogActions sx={{ flexWrap: 'wrap', rowGap: 1 }}>
          <Button disabled={busy} onClick={() => { setRegister(!register); setError(''); }}>
            {register ? 'Уже есть аккаунт' : 'Создать аккаунт'}
          </Button>
          <Button disabled={busy} onClick={dismiss}>Отмена</Button>
          <Button type="submit" variant="contained" disableElevation disabled={busy}>
            {register ? 'Зарегистрироваться' : 'Войти'}
          </Button>
        </DialogActions>
      </Box>
    </Dialog>
  );
}

function App() {
  const [session, setSession] = useState(null);
  const [user, setUser] = useState(null);
  const [accountOpen, setAccountOpen] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  async function start(reuse, knownProfile) {
    setLoading(true);
    setSession(null);
    setError('');
    try {
      const profile = knownProfile ?? await api('/api/auth');
      setUser(profile.user);
      const saved = reuse ? profile.lesson_id ?? (profile.user ? null : sessionStorage.getItem(sessionKey)) : null;
      let current = null;
      if (saved) {
        try {
          current = await api(`/api/sessions/${encodeURIComponent(saved)}`);
        } catch (failure) {
          if (failure.status !== 404) throw failure;
        }
      }
      if (!current) current = await api('/api/sessions', { method: 'POST' });
      if (!profile.user) sessionStorage.setItem(sessionKey, current.id);
      else if (sessionStorage.getItem(sessionKey) === current.id) sessionStorage.removeItem(sessionKey);
      setSession(current);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setLoading(false);
    }
  }

  async function logout() {
    setLeaving(true);
    setError('');
    try {
      const profile = await api('/api/auth/logout', { method: 'POST' });
      await start(true, profile);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setLeaving(false);
    }
  }

  useEffect(() => { void start(true); }, []);
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <header className="topbar">
        <Typography component="h1" variant="h5" sx={{ fontWeight: 600 }}>Соединение проводников</Typography>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <Button variant="outlined" size="small" disabled={loading || leaving || !session}
            onClick={() => start(false)}>Новый урок</Button>
          {user ? (
            <Button size="small" disabled={loading || leaving} onClick={logout} aria-label={`Выйти из аккаунта ${user}`}>
              Выйти
            </Button>
          ) : (
            <Button size="small" disabled={loading} onClick={() => setAccountOpen(true)}>Войти</Button>
          )}
        </Stack>
      </header>
      {error && <Alert className="screen-note" severity="error"
        action={<Button onClick={() => start(true)}>Повторить</Button>}>{error}</Alert>}
      {loading ? <p className="screen-note" role="status">Открываем урок…</p> :
        session && <Lesson key={session.id} session={session} />}
      <AccountDialog open={accountOpen} close={() => setAccountOpen(false)} onAccount={profile => {
        setAccountOpen(false);
        void start(true, profile);
      }} />
    </ThemeProvider>
  );
}

createRoot(document.getElementById('root')).render(<App />);
