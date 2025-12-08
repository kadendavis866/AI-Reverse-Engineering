'''
Input: the whole dataset (in order to get the whole vocabulary)
Output:
output_path: the input for embedding model
error_path: save all the error information in (especially when two distinct instructions map to same integer)
int2insn_map_path: the map information(int -> insn (int list))
'''
import glob
import pickle
import argparse

import insn_int
from binary_file_format import BinaryFileDef


class GetVocab(object):
    def __init__(self, config):
        self.config = config
        self.int2insn_map: dict[int, bytes] = {}
        self.get_embed_input()

    def get_embed_input(self):
        count = 0
        for file_path in glob.glob(self.config['input_folder_path'] + '/x64/*O0*.pkl', recursive=True):
            temp: BinaryFileDef = pickle.load(open(file_path, 'rb'))
            insn2int_list = []
            for func in temp.functions:
                for insn in func.inst_bytes:
                    int_value = insn_int.insn2int_inverse(insn)
                    if int_value not in self.int2insn_map:
                        self.int2insn_map[int_value] = insn
                    insn2int_list.append(str(int_value))
            with open(self.config['output_path'], 'a+') as f:
                if count == 0:
                    pass
                else:
                    f.write(' ')
                f.write(' '.join(insn2int_list))
            count += 1
        with open(self.config['int2insn_map_path'], 'wb') as f:
            #noinspection PyTypeChecker
            pickle.dump(self.int2insn_map, f)
        print(f'[embed_input] Saved the integer-insn mapping information for {count} pickles!')


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder_path', dest='input_folder_path',
                        help='The data folder saving binaries information.', type=str, default="../../data/pickles", )
    parser.add_argument('-o', '--output_path', dest='output_path',
                        help='The file saving the input for embedding model.', type=str, required=False,
                        default='embed_input')
    parser.add_argument('-m', '--int2insn_map_path', dest='int2insn_map_path',
                        help='The file saving the map information (int -> instruction (int list)).', type=str,
                        required=False, default='int2insn.map')

    args = parser.parse_args()

    config_info = {
        'input_folder_path': args.input_folder_path,
        'output_path': args.output_path,
        'int2insn_map_path': args.int2insn_map_path
    }
    return config_info


def main():
    config_info = get_config()
    GetVocab(config_info)


if __name__ == '__main__':
    main()
