lint:
	@echo "    Linting firedrake_to_pytential codebase"
	@python -m flake8 firedrake_to_pytential
	@echo "    Linting firedrake_to_pytential test suite"
	@python -m flake8 tests

THREADS=1
ifeq ($(THREADS), 1)
	PYTEST_ARGS=
else
	PYTEST_ARGS=-n $(THREADS)
endif

PYOPENCL_CTX=0
PYOPENCL_TEST=portable

test:
	@echo "   Running tests"
	@python -m pytest tests $(PYTEST_ARGS)
