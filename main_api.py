from pathlib import Path

from adapters.input.fastapi.register import register_routers
from arclith import Arclith

arclith = Arclith(Path(__file__).parent / "config.yaml")
app = arclith.fastapi()
register_routers(app, arclith)

if __name__ == "__main__":
    arclith.run_api("main_api:app")
