class BTreeNode:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.data)

a = BTreeNode(6)
b = BTreeNode(4)
c = BTreeNode(5)
d = BTreeNode(3)
e = BTreeNode(2)
f = BTreeNode(1)
g = BTreeNode(0)

a.left = b
a.right = c

b.parent = a
b.left = d
b.right = e

c.parent = a
c.left = f
# c.right = g

d.parent = b

e.parent = b

f.parent = c

# g.parent = c


def iterativeInOrder(root, func):
    if not root:
        return

    prev = None
    curr = root
    _next = None

    while curr:
        if not prev or prev.left == curr or prev.right == curr:
            if curr.left:
                _next = curr.left
            else:
                func(curr)
                _next = curr.right if curr.right else curr.parent

        elif curr.left == prev:
            func(curr)
            _next = curr.right if curr.right else curr.parent

        else:
            _next = curr.parent

        prev = curr
        curr = _next

def iterativePreOrder(root, func):
    if not root:
        return

    prev = None
    curr = root
    _next = None

    while curr:
        if not prev or prev.left == curr or prev.right == curr:
            func(curr)
            if curr.left:
                _next = curr.left
            else:
                _next = curr.right if curr.right else curr.parent

        elif curr.left == prev:
            _next = curr.right if curr.right else curr.parent

        else:
            _next = curr.parent

        prev = curr
        curr = _next

def iterativePostOrder(root, func):
    if not root:
        return

    prev = None
    curr = root
    _next = None

    while curr:
        print(curr, curr.left, curr.right)
        if not prev or prev.left == curr or prev.right == curr:
            # traverse left
            if curr.left:
                _next = curr.left
                print("left", _next)
            elif curr.right:
                # if there is nothing on the left of the node
                print('here')
                _next = curr.right
            else:
                func(curr, "leave node")
                _next = curr.parent

        elif curr.left == prev:
            # traverse right
            if curr.right:
                _next = curr.right
                print("right", _next)
            else:
                # if right is empty return to parent
                # print('here')
                # func(prev)
                func(curr)
                _next = curr.parent

        else:
            # root node
            func(curr)
            _next = curr.parent

        prev = curr
        curr = _next

# iterativeInOrder(a, print)   # 3 4 2 6 1 5
# iterativePreOrder(a, print)  # 6 4 3 2 5 1
iterativePostOrder(a, print) # 3 2 4 1 5 6
