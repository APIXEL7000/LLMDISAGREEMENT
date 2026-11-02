"""Check your setup.   python scripts/check_setup.py [--live --gemini-model NAME --ollama-model NAME]
Never prints key values.  --live sends one tiny request to each model you name (uses a little quota)."""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config                                        # noqa: E402
from src.llm_clients import GenRequest, LLMError, list_ollama_models, make_client, ollama_base  # noqa: E402
from src.secrets import get_key, load_dotenv                          # noqa: E402

ok = True


def line(passed, msg, fix=""):
    global ok
    ok &= passed
    print(("PASS  " if passed else "FAIL  ") + msg + (f"\n      -> {fix}" if fix and not passed else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--gemini-model")
    ap.add_argument("--ollama-model")
    a = ap.parse_args()
    line(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (need 3.10+)", "install a newer Python")
    for pkg in ("numpy", "pandas", "matplotlib"):
        try:
            importlib.import_module(pkg); line(True, f"package {pkg}")
        except ImportError:
            line(False, f"package {pkg}", "run: pip install -r requirements.txt")
    env = ROOT / ".env"
    line(env.is_file(), ".env file found", "copy .env.example to .env and add your key(s)")
    load_dotenv()
    if (ROOT / ".git").exists():
        try:
            ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT, stderr=subprocess.DEVNULL).returncode == 0
        except OSError:
            ignored = False
        line(ignored, ".env is excluded from Git", "add '.env' to .gitignore")
    else:
        line(".env" in (ROOT / ".gitignore").read_text(), ".env is listed in .gitignore (no git repo here yet)", "add '.env' to .gitignore")
    print("INFO  Gemini key configured: " + ("yes" if get_key("gemini") else "no  (set GEMINI_API_KEY in .env)"))
    try:
        models = list_ollama_models()
        line(True, f"Ollama reachable at {ollama_base()}; installed models: {', '.join(models) or '(none yet)'}")
        if not models:
            print("      -> install one with: ollama pull <model name>   (browse ollama.com/library)")
    except LLMError as exc:
        line(False, "Ollama reachable", str(exc))
    if a.live:
        for spec in filter(None, [a.gemini_model and f"gemini:{a.gemini_model}", a.ollama_model and f"ollama:{a.ollama_model}"]):
            try:
                r = make_client(spec, Config()).generate(GenRequest("t", "Reply with the single word: OK", 0, 0.0, 16))
                line(True, f"{spec} replied: {r.text.strip()[:40]!r}")
            except LLMError as exc:
                line(False, spec, str(exc))
    print("\nAll checks passed." if ok else "\nSome checks failed - see the fixes above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
