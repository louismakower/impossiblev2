install uv and sync
`curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync`

run `sudo usermod -aG docker $USER` to add the user to the docker group. then login/logout of the shell (or delete ~/.vscode-server) to reload
## 100-epoch runs

The main-result logs, 100 epochs of every context × grading cell, one log per
model × variant (`logs/` is gitignored, so this is the record of which files
they are):

backdoor (`logs/backdoor/`)
- kimi-k3 baseline: `2026-09-15T11-02-21-00-00_backdoor-hosted_NvsvWc8fAh5kRndMKY75MC.eval`
- kimi-k3 realistic: `2026-09-15T11-02-22-00-00_backdoor-hosted_ArXuntApbFksAEymLzz2j2.eval`
  (status `error`: the run died with `OSError: Argument list too long` after
  1101 scored samples; 50 more were cancelled unscored, and 48 never started)
- glm-5.3 baseline: `2026-09-16T09-25-36-00-00_backdoor-hosted_VSKdyr3EDUddxBGhBaVabs.eval`
- glm-5.3 realistic: `2026-09-16T17-17-37-00-00_backdoor-hosted_L9ButorJpV6UX4V2QrkYi6.eval`
  (1193 of 1200 samples)

test_train (`logs/test_train/`)
- kimi-k3 baseline: `2026-09-16T15-49-30-00-00_test-train_TfV4JMhTxtMVpgMX87DgLi.eval`
- kimi-k3 realistic: `2026-09-16T15-49-30-00-00_test-train_AKAjS9Qh6hSuR7ptooqneK.eval`
- glm-5.3 baseline: `2026-09-16T15-49-31-00-00_test-train_akAYq7jmhyibCWWzHaC8F4.eval`
- glm-5.3 realistic: `2026-09-16T15-49-31-00-00_test-train_A9XYQK8eXukCs4QszH5sxm.eval`

The earlier same-named `2026-09-15T11-02-22` / `2026-09-15T11-02-23` glm
backdoor logs are the errored first attempts at the two glm runs, and the
`2026-09-16T14-54` test_train logs are cancelled false starts.

All eight are awareness-scanned (`logs/<task>/scans`). Build the awareness
comparison page with:

    uv run common/awareness.py <the eight logs above> --out plots/awareness.html
