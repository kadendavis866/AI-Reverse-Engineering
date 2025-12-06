class BinaryFileDef:
    class Function:
        name: str
        num_args: int
        args_type: list[str]
        ret_type: str
        inst_strings: list[str]
        inst_bytes: list[bytes]
        boundaries: tuple[int, int]

    class Structure:
        name: str
        fields: list[str]

    class FunctionCalls:
        class FunctionCall:
            caller: str
            call_instr_indices: list[int]

        name: str
        calls: list[FunctionCall]

    class ExternFunction:
        name: str
        address: int

    binary_filename: str
    arch: str
    text_addr: int
    structures: list[Structure]
    functions: list[Function]
    function_calls: list[FunctionCalls]
    binRawBytes: bytes
    extern_functions: list[ExternFunction]