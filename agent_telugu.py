"""
Telugu Voice Agent — run with: python agent_telugu.py dev
"""
from agent_base import make_entrypoint
from livekit.agents import WorkerOptions, cli

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=make_entrypoint("telugu"),
        worker_type="room",
        agent_name="telugu-agent",
    ))
