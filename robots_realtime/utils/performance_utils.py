import logging
import os

logger = logging.getLogger(__name__)


def set_realtime_and_pin(
    pinned_cpu: int,
    realtime_priority: int = 90,
    require_realtime: bool = False,
) -> None:
    # Set CPU affinity to pin to specific core
    os.sched_setaffinity(0, {pinned_cpu})
    logger.info(f"Pinned robot process to CPU core {pinned_cpu}")

    # Set real-time scheduling policy (SCHED_RR) with high priority
    # We use SCHED_RR instead of SCHED_FIFO because otherwise the saving operation
    # will block the motor control loop, causing the robot arm to die.
    param = os.sched_param(int(realtime_priority))
    try:
        os.sched_setscheduler(0, os.SCHED_RR, param)
    except PermissionError:
        msg = (
            f"Could not set SCHED_RR priority {realtime_priority}; "
            "continuing with CPU affinity only. Grant rtprio/CAP_SYS_NICE "
            "or run with appropriate privileges to enable realtime scheduling."
        )
        if require_realtime:
            raise PermissionError(msg) from None
        logger.warning(msg)
    else:
        logger.info(f"Set SCHED_RR scheduling with priority {realtime_priority}")
