install uv and sync
`curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync`

run `sudo usermod -aG docker $USER` to add the user to the docker group. then login/logout of the shell (or delete ~/.vscode-server) to reload