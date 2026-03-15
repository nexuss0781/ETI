import os
import subprocess
import requests
import sys

# ==========================
# USER CONFIG
# ==========================
GITHUB_TOKEN = "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"  # revoke after testing
GITHUB_USER = "nexuss0781"
GIT_EMAIL = "nexuss0781@gmail.com"

DEFAULT_BRANCH = "main"
DEFAULT_COMMIT_MSG = "Update"

GITHUB_API = "https://api.github.com"

# ==========================
# HELPERS
# ==========================
def run(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"❌ Command failed: {cmd}")

def output(cmd):
    return subprocess.getoutput(cmd).strip()

def has_git():
    return os.path.isdir(".git")

def has_origin():
    return "origin" in output("git remote")

# ==========================
# MAIN
# ==========================
def main():
    # --------------------------
    # INIT GIT IF NEEDED
    # --------------------------
    if not has_git():
        print("📁 Initializing git")
        run("git init")

    # --------------------------
    # CONFIG USER (ALWAYS SAFE)
    # --------------------------
    run(f'git config user.name "{GITHUB_USER}"')
    run(f'git config user.email "{GIT_EMAIL}"')

    # --------------------------
    # SETUP ORIGIN (ONE TIME)
    # --------------------------
    if not has_origin():
        choice = input("New or existing repo? (new/existing): ").strip().lower()
        if choice not in ("new", "existing"):
            sys.exit("❌ Choose 'new' or 'existing'")

        repo_name = input("Repository name: ").strip()
        if not repo_name:
            sys.exit("❌ Repo name required")

        repo_url = f"https://github.com/{GITHUB_USER}/{repo_name}.git"

        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }

        if choice == "new":
            private = input("Private repo? (yes/no) [no]: ").lower() in ("yes", "y")

            r = requests.post(
                f"{GITHUB_API}/user/repos",
                headers=headers,
                json={"name": repo_name, "private": private},
            )

            if r.status_code != 201:
                sys.exit(f"❌ Repo creation failed: {r.status_code} {r.text}")

            print("✅ GitHub repo created")

        print("🔗 Setting origin")
        run(f"git remote add origin {repo_url}")

    # --------------------------
    # COMMIT & PUSH (EVERY RUN)
    # --------------------------
    commit_msg = input(f"Commit message [{DEFAULT_COMMIT_MSG}]: ").strip() or DEFAULT_COMMIT_MSG
    branch = input(f"Branch [{DEFAULT_BRANCH}]: ").strip() or DEFAULT_BRANCH

    run("git add .")
    run(f'git commit -m "{commit_msg}" --allow-empty')
    run(f"git branch -M {branch}")
    run(f"git push -u origin {branch}")

    print("🚀 Push complete")

if __name__ == "__main__":
    main()
