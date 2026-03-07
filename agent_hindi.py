"""
Hindi Voice Agent — run with: python agent_hindi.py dev
"""
from agent_base import make_entrypoint
from livekit.agents import WorkerOptions, cli

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=make_entrypoint("hindi"),
        worker_type="room",
        agent_name="hindi-agent",
    ))
