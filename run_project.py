import os
import sys


def run():
    # Ensure outputs directory exists for notebook saves
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(project_root, 'outputs', 'plots'), exist_ok=True)

    # Fix asyncio policy on Windows for nbconvert/zmq compatibility
    if sys.platform.startswith('win'):
        try:
            import asyncio
            from asyncio import WindowsSelectorEventLoopPolicy
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    cmd = (
        'jupyter nbconvert --to notebook --execute '
        'notebooks/Traffic_Pollution_EDA_and_Modeling.ipynb '
        '--output Traffic_Pollution_EDA_and_Modeling.out.ipynb '
        '--output-dir notebooks'
    )
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise SystemExit(f"Notebook execution failed with code {exit_code}")


if __name__ == "__main__":
    run()
