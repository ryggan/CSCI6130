import brainfuck
import compiler
import reward
from utils import prepare_list

# Input list
unsorted_list = [9, 1, 6, 3, 2]

code = prepare_list(unsorted_list)
code = compiler.read(code)

output = str(compiler.eval(code))
output = list(map(lambda x: int(x), output))

print("Reward: " + str(reward.evaluate_sorting(output)))
print("Unsorted: " + str(unsorted_list))
print("Output: " + str(output))
unsorted_list.sort()
print("Expeced: " + str(unsorted_list))
