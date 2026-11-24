"""Regressions for the independent submission audit; no remote provider."""
import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from app.answers import resolve_ordinal_elements
from app.claims import assess_solution, evidence_chunks, literal_overall, solution_schema, sourced_solution
from app.knowledge import CASES, CIRCUITS, question
from app.main import create_app
from app.tutor import new_state, respond

class MethodFlowTests(unittest.TestCase):
    def test_proposal_comparison_and_reload_without_skipping_operations(self):
        with tempfile.TemporaryDirectory() as folder:
            app = create_app(Path(folder) / 'test.sqlite3', type('Offline', (), {'enabled': False})())
            with TestClient(app) as client:
                session = client.post('/api/sessions').json()
                url = '/api/sessions/' + session['id']
                def send(text):
                    nonlocal session
                    session = client.post(url + '/turns', json={
                        'message_id': str(session['version']), 'version': session['version'], 'text': text,
                    }).json()
                    return session
                send('R1 и R2 сверху, R3 снизу')
                self.assertEqual(session['state']['phase'], 'method')
                self.assertNotIn('1. Выделить', session['message']['text'])
                self.assertEqual(client.get(url).json()['state']['phase'], 'method')
                send('Сначала найду элементы и узлы.')
                self.assertEqual(session['state']['phase'], 'method_compare')
                self.assertIn('отсутствие ответвлений', session['message']['text'])
                before = copy.deepcopy(session['state'])
                send('Что такое узел?')
                self.assertEqual(session['state'], before)
                self.assertIn('непрерывным проводом', session['message']['text'])
                self.assertEqual(client.get(url).json()['state']['phase'], 'method_compare')
                send('Добавлю проверку ответвлений и общий вывод.')
                self.assertEqual(session['state']['phase'], 'guided')
                self.assertEqual(session['state']['step_label'], 'Элементы')
                _, stored, _, _ = app.state.store.load(url.rsplit('/', 1)[1], None)
                self.assertEqual(stored['method_proposal'], 'Сначала найду элементы и узлы.')
                self.assertEqual(stored['method_comparison'], 'Добавлю проверку ответвлений и общий вывод.')
                self.assertTrue(stored['method_completed'])

    def test_difficulty_has_an_explicit_adoption_path(self):
        state = new_state()
        state['phase'] = 'method'
        offline = type('Offline', (), {'enabled': False})()
        asyncio.run(respond(state, 'не знаю', offline))
        self.assertEqual(state['phase'], 'method_compare')
        asyncio.run(respond(state, 'Использовать образец', offline))
        self.assertEqual((state['phase'], state['operation']), ('guided', 0))

class ExtractionAuditTests(unittest.TestCase):
    def test_wrong_overall_enum_cannot_override_literal_conclusion(self):
        series = 'R1 и R2 соединены последовательно, в C нет ответвлений'
        parallel = 'Ветвь R1–R2 и R3 параллельны между A и B'
        for declared, slot in (('Параллельное', 'mixed'), ('Смешанное', 'parallel'), ('Последовательное', 'mixed')):
            text = declared + '. ' + series + '. ' + parallel
            solution = sourced_solution({'overall': slot, 'overall_evidence': declared, 'relations': [
                {'relation': 'series', 'groups': ['R1', 'R2'], 'nodes': ['C'], 'no_extra_branch': True, 'evidence': series},
                {'relation': 'parallel', 'groups': ['R1+R2', 'R3'], 'nodes': ['A', 'B'], 'no_extra_branch': False, 'evidence': parallel},
            ]}, text, CIRCUITS['C01'])
            status, _ = assess_solution('C01', question(CASES['C01'], 'overall'), solution, text)
            self.assertEqual(status, 'complete' if declared == 'Смешанное' else 'contradiction')
        for ambiguous in ('Не параллельное', 'Если смешанное', 'Последовательное или смешанное'):
            self.assertIsNone(literal_overall(ambiguous))

    def test_number_words_allow_only_existing_component_evidence(self):
        text = 'На верхней дорожке резисторы номер один и номер два, на нижней — номер три.'
        chunks = evidence_chunks(text)
        schema = solution_schema('branches', ['R1', 'R2', 'R3'], ['A', 'B', 'C'], chunks)
        for element in ('R1', 'R2', 'R3'):
            allowed = schema['properties']['placements']['properties'][element]['properties']['evidence']['enum']
            self.assertTrue(any(chunks[key] == text.strip('.') for key in allowed if key))
        self.assertNotIn('r3', resolve_ordinal_elements('номер три', CIRCUITS['C03'].components))
        self.assertEqual(resolve_ordinal_elements('второй вывод', CIRCUITS['C01'].components), 'второй вывод')
