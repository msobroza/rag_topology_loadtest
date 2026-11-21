# Where doclang's core is, read by the repository's Makefile and the toolkit's
# (include doclang.mk, include ../doclang.mk), so that the two cannot say it
# differently.
#
# doclang's core (doclang-domain-core), which the core depends on, is private and
# on no index: it is installed from a clone of document_doclang, as its working
# tree stands. The clone sits beside the repository's main checkout, found from
# git's common directory, which is the main checkout's .git from a worktree too;
# DOCLANG_CLONE names another place. Expanded when a target that needs it runs,
# so no other target needs the clone. An empty DOCLANG_CLONE=, in the shell or
# on the command line, is read as unset, not as a clone in no directory: hence
# the test and override rather than ?=, which keeps a variable defined empty.
ifeq ($(strip $(DOCLANG_CLONE)),)
override DOCLANG_CLONE = $(abspath $(dir $(shell git rev-parse --path-format=absolute --git-common-dir 2>/dev/null))..)/document_doclang
endif
DOCLANG_CORE = $(DOCLANG_CLONE)/packages/core
# Its directory, or the target stops in one line, under make -n too.
DOCLANG_CORE_OR_STOP = $(if $(wildcard $(DOCLANG_CORE)/pyproject.toml),$(DOCLANG_CORE),$(error no clone of document_doclang in $(DOCLANG_CLONE): clone it there, or name its directory with DOCLANG_CLONE))
