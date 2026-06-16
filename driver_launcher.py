import os
import driver

# Cleanup old flag files before starting
for flag_file in ["color.flag", "depth.flag", "all_shutdown.flag"]:
    if os.path.exists(flag_file):
        os.remove(flag_file)

driver.sender((720, 1280, 3))
