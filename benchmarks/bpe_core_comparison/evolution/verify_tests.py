"""Record a complete test run; kept as a file so spawn workers can import it."""
import io
import json
from pathlib import Path
import time
import unittest

HERE = Path(__file__).resolve().parent


def main():
    stream = io.StringIO()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    record = dict(tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), skipped=len(result.skipped),
                  elapsed_seconds=time.perf_counter()-start,
                  successful=result.wasSuccessful(), output=stream.getvalue())
    (HERE/'unit-tests.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
