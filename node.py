class Node:
    def __init__(self,item):
        self.data = item
        self.parent = None
    def PrintTree(self):
        print(self.data)
        