import json
import re
import sys
from pathlib import Path
from uuid import uuid4
import httpx

base = 'https://fizika.storozhq.ru'
rows = []
with httpx.Client(base_url=base, timeout=45) as client:
    health = client.get('/api/health'); health.raise_for_status()
    assert health.json() == {'ok': True, 'gigachat_configured': True}
    page = client.get('/'); page.raise_for_status()
    assets = re.findall(r'["\'](/assets/[^"\']+)["\']', page.text)
    assert assets
    for asset in assets:
        client.get(asset).raise_for_status()
    response = client.post('/api/sessions'); response.raise_for_status()
    session = response.json()
    sid = session['id']; route = f'/api/sessions/{sid}/turns'
    assert session['state']['phase'] == 'trial'
    def turn(text):
        global session
        before = session
        body = {'message_id': uuid4().hex, 'version': session['version'], 'text': text}
        response = client.post(route, json=body); response.raise_for_status()
        session = response.json()
        saved = client.get(f'/api/sessions/{sid}'); saved.raise_for_status()
        assert saved.json()['state'] == session['state']
        duplicate = client.post(route, json=body); duplicate.raise_for_status()
        assert duplicate.json() == session
        rows.append({'text': text, 'reply': session['message']['text'],
                     'before': before['state'], 'after': session['state'],
                     'grounding': session['message'].get('grounding'),
                     'post_get_equal': True, 'idempotent': True})
        Path(sys.argv[1]).write_text(json.dumps({'target':base,'turns':rows,'pass':False},ensure_ascii=False,indent=2))
        return session
    turn('Привет ещё раз! Ветку сверху составляют р1 вместе с р2; под ними идёт р3.')
    assert session['state']['phase'] == 'guided'
    assert session['message']['text'].startswith('Верно.')
    assert session['message']['grounding']['status'] == 'matched'
    nodes_question = session['state']['question']
    turn('А что такое резистор?')
    assert session['state']['question'] == nodes_question
    assert 'Резистор —' in session['message']['text']
    turn('R1: C–A; R2: B–C; R3: B–A. А буква R обозначает резистор?')
    assert session['state']['question'] != nodes_question
    assert session['message']['text'].startswith('Верно.')
    stale = client.post(route, json={'message_id': uuid4().hex, 'version': 0, 'text': 'Подтвердить'})
    assert stale.status_code == 409
    saved = client.get(f'/api/sessions/{sid}'); saved.raise_for_status()
    assert saved.json()['state'] == session['state']
report = {'target': base, 'health': health.json(), 'assets_ok': True, 'stale_version_rejected': True,
          'turns': rows, 'pass': True}
Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('Public HTTPS, GigaChat dialogue, explicit correct feedback, term question, saved progress, idempotency and version conflict: PASS')
