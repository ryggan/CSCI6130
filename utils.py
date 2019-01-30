def prepare_list(data):
    # Prepare our Brainfuck code
    code = ""
    for digit in data:
        for _ in range(ord(str(digit))):
            code += "+"
        code += ">"

    # Move pointer to beginning
    for _ in range(len(data) + 1):
        code += "<"

    return code
