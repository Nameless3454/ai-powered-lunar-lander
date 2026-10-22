
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

def run_script(script_name, extra_args=None):
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        print(f"\nError: {script_name} was not found.")
        input("Press Enter to continue...")
        return

    command = [sys.executable, str(script_path)]
    if extra_args:
        command.extend(extra_args)

    print(f"\nStarting {script_name}...\n")
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=False)
    except KeyboardInterrupt:
        print("\nExecution interrupted.")
    except Exception as error:
        print(f"\nError: {error}")

def main():
    while True:
        print("\n" + "=" * 42)
        print("       ROCKET R1 COMMAND DASHBOARD")
        print("=" * 42)
        print("1. Run visual simulation")
        print("2. Run headless evaluation (100 episodes)")
        print("3. Exit")
        print("=" * 42)

        choice = input("Select an option [1-3]: ").strip()

        if choice == "1":
            run_script("simulate_scene.py")

        elif choice == "2":
            # Change this to the script that contains
            # your normal, non-UI evaluation loop.
            run_script("evaluate.py", ["--episodes", "100"])

        elif choice == "3":
            print("Dashboard closed.")
            break

        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
