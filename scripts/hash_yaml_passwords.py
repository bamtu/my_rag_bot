"""Hash any plaintext passwords in app/auth_config.yaml in place.

Run once after editing auth_config.yaml with plaintext passwords.
Entries that already start with '$2' (bcrypt) are left untouched.

Usage:
    PYTHONPATH=. python scripts/hash_yaml_passwords.py
"""

from pathlib import Path

import bcrypt
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "app" / "auth_config.yaml"


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    users = cfg["credentials"]["usernames"]
    changed = []
    for username, info in users.items():
        pw = info["password"]
        if isinstance(pw, str) and pw.startswith("$2"):
            continue
        hashed = bcrypt.hashpw(str(pw).encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        info["password"] = hashed
        changed.append(username)

    if not changed:
        print("No plaintext passwords found. Nothing to do.")
        return

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Hashed passwords for: {', '.join(changed)}")
    print(f"Updated: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
