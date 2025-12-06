import pickle
import os

from binary_file_format import BinaryFileDef

INPUT_DIR = "data/pickles_old/x64"
OUTPUT_DIR = "data/pickles/x64"


def get_file_paths():
    cur_paths = [INPUT_DIR]
    paths = []
    while len(cur_paths) > 0:
        path = cur_paths.pop()
        if not os.path.isdir(path) and (path[-4:] == '.pkl'):
            paths.append(path)
        elif os.path.isdir(path):
            for file in os.listdir(path):
                cur_paths.append(os.path.join(path, file))
        else:
            pass
    return paths


def convert_pickle(input_path: str, output_path: str):
    p: dict = pickle.load(open(input_path, "rb"), encoding='latin1')

    bf = BinaryFileDef()
    bf.binary_filename = p['binary_filename']
    bf.arch = p['arch']
    bf.text_addr = int(p['text_addr'], 16)
    bf.binRawBytes = bytes(p['bin_raw_bytes'], 'utf-8')
    bf.structures = []
    for struct_name, fields in p['structures'].items():
        struct = BinaryFileDef.Structure()
        struct.name = struct_name
        struct.fields = fields
        bf.structures.append(struct)
    bf.functions = []
    for func_name, func_info in p['functions'].items():
        func = BinaryFileDef.Function()
        func.name = func_name
        func.num_args = func_info['num_args']
        func.args_type = func_info['args_type']
        func.ret_type = func_info['ret_type']
        func.inst_strings = func_info['inst_strings']
        func.inst_bytes = [bytes(byte_list) for byte_list in func_info['inst_bytes']]
        func.boundaries = func_info['boundaries']
        bf.functions.append(func)
    bf.function_calls = []
    for fc_name, calls in p['function_calls'].items():
        func_calls = BinaryFileDef.FunctionCalls()
        func_calls.name = fc_name
        func_calls.calls = []
        for call_info in calls:
            func_call = BinaryFileDef.FunctionCalls.FunctionCall()
            func_call.caller = call_info['caller']
            func_call.call_instr_indices = call_info['call_instr_indices']
            func_calls.calls.append(func_call)
        bf.function_calls.append(func_calls)
    bf.extern_functions = []
    for extern_name, address in p['extern_functions'].items():
        extern_func = BinaryFileDef.ExternFunction()
        extern_func.name = extern_name
        extern_func.address = address
        bf.extern_functions.append(extern_func)

    # noinspection PyTypeChecker
    pickle.dump(bf, open(output_path, "wb"))


def main():
    paths = get_file_paths()
    for path in paths:
        convert_pickle(path, os.path.join(OUTPUT_DIR, os.path.basename(path)))


if __name__ == "__main__":
    main()
