"""Tests run with the Simplified Chinese source texts, whatever the system language (1.5.1)."""
import os

os.environ['LUMEN_LANGUAGE'] = os.environ.get('LUMEN_TEST_LANGUAGE', 'zh_CN')
