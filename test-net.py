# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import sys


class StdoutToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip():  # Skip empty lines
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass


# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel("WARNING")
stdout_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stdout_handler)

# Redirect stdout to the logger with a specific log level (e.g., WARNING)
old_out = sys.stdout
sys.stdout = StdoutToLogger(logger, level=logging.INFO)

# Now, any print statements will be captured by the logger with a level of WARNING
print("This will be captured as a WARNING log message.")

logger.warning("hello")

import logging

import pypsa

logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.ERROR)
pypsa.opf.logger.setLevel(logging.ERROR)
pypsa.optimization.optimize.logger.setLevel(logging.ERROR)
network = pypsa.examples.ac_dc_meshed(from_master=True)
network.optimize()

sys.stdout = old_out

print("finished")
