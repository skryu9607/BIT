import math
from env import Node
a = Node(1,2,3)
b = Node(2,3,4)

print(math.hypot(a.x-b.x,a.y-b.y,a.z-b.z))
