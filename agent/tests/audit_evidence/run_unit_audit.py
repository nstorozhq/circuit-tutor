"""Run the same six offline audit modules under the portable import guards."""
import sys
import unittest
import api_regressions  # Sets temporary TUTOR_DB and suppresses dotenv at import.

MODULES = (
    "tests.test_dialogue", "tests.test_rag", "tests.test_lesson",
    "tests.test_mixed_turns", "tests.test_claims", "tests.test_audit_regressions",
)

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    sys.exit(not result.wasSuccessful())
