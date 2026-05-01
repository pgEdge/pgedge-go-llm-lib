#-------------------------------------------------------------------------
#
# pgedge-go-llm-lib
#
# Copyright (c) 2025 - 2026, pgEdge, Inc.
# This software is released under The PostgreSQL License
#
#-------------------------------------------------------------------------

# Default Go toolchain. Override with `make GO=/path/to/go test`.
GO ?= go

# golangci-lint binary. The Makefile prefers a pinned version installed
# under $(GOBIN) but falls back to whatever is on $PATH.
GOLANGCI_LINT ?= $(shell command -v golangci-lint 2>/dev/null)
GOLANGCI_LINT_VERSION ?= v1.62.2

PKGS := ./...

.PHONY: all
all: fmt vet test

.PHONY: build
build:
	$(GO) build $(PKGS)

.PHONY: test
test:
	$(GO) test -count=1 -race $(PKGS)

.PHONY: test-short
test-short:
	$(GO) test -count=1 -short $(PKGS)

.PHONY: coverage
coverage:
	$(GO) test -count=1 -coverprofile=coverage.out -covermode=atomic $(PKGS)
	$(GO) tool cover -func=coverage.out | tail -1
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "HTML coverage report: coverage.html"

.PHONY: bench
bench:
	$(GO) test -bench=. -benchmem -run=^$$ $(PKGS)

.PHONY: fmt
fmt:
	$(GO) fmt $(PKGS)
	gofmt -w -s .

.PHONY: fmt-check
fmt-check:
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt: the following files need formatting:"; \
		echo "$$unformatted"; \
		exit 1; \
	fi

.PHONY: vet
vet:
	$(GO) vet $(PKGS)

.PHONY: lint
lint: fmt-check vet
	@if [ -z "$(GOLANGCI_LINT)" ]; then \
		echo "golangci-lint is not installed; run 'make lint-install' first."; \
		exit 1; \
	fi
	$(GOLANGCI_LINT) run $(PKGS)

.PHONY: lint-install
lint-install:
	$(GO) install github.com/golangci/golangci-lint/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

.PHONY: tidy
tidy:
	$(GO) mod tidy

.PHONY: docs
docs:
	@command -v mkdocs >/dev/null 2>&1 || { echo "mkdocs is not installed (pip install mkdocs-material)"; exit 1; }
	mkdocs serve

.PHONY: docs-build
docs-build:
	@command -v mkdocs >/dev/null 2>&1 || { echo "mkdocs is not installed (pip install mkdocs-material)"; exit 1; }
	mkdocs build --strict

.PHONY: clean
clean:
	rm -f coverage.out coverage.html

.PHONY: help
help:
	@echo "pgedge-go-llm-lib — common targets:"
	@echo "  make build         compile all packages"
	@echo "  make test          run tests with -race"
	@echo "  make test-short    run tests with -short (skips slow tests)"
	@echo "  make coverage      generate coverage.out + coverage.html"
	@echo "  make bench         run benchmarks"
	@echo "  make fmt           run gofmt -s -w on all .go files"
	@echo "  make fmt-check     fail if any file needs gofmt"
	@echo "  make vet           run go vet"
	@echo "  make lint          run gofmt-check, go vet, and golangci-lint"
	@echo "  make lint-install  install golangci-lint at the pinned version"
	@echo "  make tidy          run go mod tidy"
	@echo "  make docs          serve MkDocs site locally"
	@echo "  make docs-build    build MkDocs site (strict)"
	@echo "  make clean         remove generated files"
