import logging
from datetime import datetime
from pathlib import Path


_CONFIGURED_LOGS: dict[str, Path] = {}


def configure_logging(name: str, output_dir: str | Path = "output") -> tuple[logging.Logger, Path]:
    """Configure console and timestamped file logging under output_dir/logs."""
    log_dir = (Path(output_dir) / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    configured_path = _CONFIGURED_LOGS.get(name)
    if configured_path is not None and configured_path.parent == log_dir:
        return logger, Path(configured_path)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{name}_{timestamp}.log"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    _CONFIGURED_LOGS[name] = log_path
    return logger, log_path
