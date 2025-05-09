import random
import time

def get_current_wait_time(hospital: str) -> int | str:
    """Dummy function to generate fake wait times"""

    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist."
    time.sleep(1)  # Simulate a delay in getting the wait time
    return random.randint(1, 10000)  # Random wait time between 1 and 120 minutes




