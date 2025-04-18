import os
import sys
import pandas as pd
import pynvml as N

MB = 1024 * 1024

def get_usage(device_index):
    N.nvmlInit()

    handle = N.nvmlDeviceGetHandleByIndex(device_index)

    

    usage = [{'pid': nv_process.pid, 'memory': nv_process.usedGpuMemory // MB} for nv_process in
             N.nvmlDeviceGetComputeRunningProcesses(handle) + N.nvmlDeviceGetGraphicsRunningProcesses(handle)]

    # if len(usage) == 1:
    #     usage = usage[0]
    # else:
    #     raise KeyError("PID not found")

    # return usage
    return pd.DataFrame(usage)

if __name__ == "__main__":
   print(get_usage(int(sys.argv[1])))
