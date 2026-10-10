"""Tests run with the Simplified Chinese source texts, whatever the system language (1.5.1).

1.6.0: tests use their own rating catalog and control-channel name, so they never touch the
user's library and never talk to a LUMEN RAW window the user has open."""
import os
import tempfile

os.environ['LUMEN_LANGUAGE'] = os.environ.get('LUMEN_TEST_LANGUAGE', 'zh_CN')
os.environ['LUMEN_CATALOG'] = os.path.join(tempfile.mkdtemp(prefix='lumen-test-catalog-'), 'catalog.jsonl')
os.environ['LUMEN_CONTROL_NAME'] = f'lumen-raw-test-{os.getpid()}'
