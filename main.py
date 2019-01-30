import brainfuck
import compiler
from utils import prepare_list

# Input list
unsorted_list = [9, 1, 6, 3]
sorted_list = unsorted_list.sort()

code = prepare_list(unsorted_list)

# Print the content of every
for _ in range(len(unsorted_list)):
    code += ".>"

code = compiler.read(code)

output = str(compiler.eval(code))
output = list(map(lambda x: int(x), output))

print(unsorted_list)
print(output)
