"""Test configuration: keep the test database out of the project folder."""

import os
import tempfile

# Must run before smartbizlq.db.session is imported, because it creates the engine at import time.
os.environ["SQLITE_FILE"] = os.path.join(tempfile.mkdtemp(prefix="smartbizlq_tests_"), "test.db")
for var in ("DB_USER", "DB_PASS", "DB_PASSWORD", "DB_NAME", "ADMIN_EMAIL"):
    os.environ.pop(var, None)
