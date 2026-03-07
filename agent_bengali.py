"""
Bengali Voice Agent — run with: python agent_bengali.py dev
"""
from agent_base import make_entrypoint
from livekit.agents import WorkerOptions, cli

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=make_entrypoint("bengali"),
        worker_type="room",
        agent_name="bengali-agent",
    ))
