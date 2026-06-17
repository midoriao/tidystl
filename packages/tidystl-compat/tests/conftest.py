"""Pytest setup for the tidystl_compat test suite.

Registering the compat package here guarantees the backends are resolvable by
name in every test module -- including when a single test file is run in
isolation -- rather than relying on some other test module having imported the
package first.
"""

import tidystl
import tidystl_compat

tidystl.use(tidystl_compat)
