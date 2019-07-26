.PHONY: lint test_meshes example_meshes meshes test

lint:
	@echo "    Linting firedrake_to_pytential codebase"
	@python -m flake8 firedrake_to_pytential
	@echo "    Linting firedrake_to_pytential test suite"
	@python -m flake8 tests

test_meshes:
	@echo "    Building test meshes"
	@python3 bin/make_meshes test
	@echo "	   Test meshes built"

example_meshes:
	@echo "    Building example meshes"
	@python3 bin/make_meshes example
	@echo "	   Example meshes built"

meshes: test_meshes example_meshes

THREADS=1
ifeq ($(THREADS), 1)
	PYTEST_ARGS=
else
	PYTEST_ARGS=-n $(THREADS)
endif

PYOPENCL_TEST=portable

test:
	@make test_meshes
	@echo "   Running tests"
	@python -m pytest tests $(PYTEST_ARGS)
