# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.

import os
import argparse

import pycuda.driver as cuda
import pycuda.autoinit  # noqa

import tensorrt as trt


def GiB(val):
    return val * 1 << 30


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--datadir",
                        help="Location of the TensorRT sample data directory.",
                        default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
