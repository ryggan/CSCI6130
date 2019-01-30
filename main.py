import brainfuck
import compiler

# Input list
unsorted_list = [9, 1, 6, 3]
sorted_list = unsorted_list.sort()

# Prepare our Brainfuck code
code = ""
for digit in unsorted_list:
    for _ in range(ord(str(digit))):
        code += "+"
    code += ">"

# Move pointer to beginning
for _ in range(len(unsorted_list) + 1):
    code += "<"

# Print the content of every
for _ in range(len(unsorted_list)):
    code += ".>"

code = compiler.read(code)

output = str(compiler.eval(code))
output = list(map(lambda x: int(x), output))

print(unsorted_list)
print(output)
