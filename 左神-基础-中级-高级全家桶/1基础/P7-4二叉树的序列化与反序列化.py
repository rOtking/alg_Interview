'''

二叉树唯一的字符串结构！
'''
'''
用什么方式遍历都可以，就是在叶子结点左右None的时候加入一个标识如'#'即可。

注意
消费queue的队列是不断在缩减的！


重点看！
'''
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        # todo 为了区分，加入'_'    不然混在一起区分不了  不是为了区分#，是为了区分数字   如1和2  12  1_2
        if root is None:
            return '#_'

        res = str(root.val) + '_'
        res += self.serialize(root.left)
        res += self.serialize(root.right)
        return res

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        # 变成list好处理
        data1 = data.split('_')
        queue = []
        for x in data1:
            if x != '':
                queue.append(x)
        root = self.process(queue)
        return root

    def process(self, queue):
        cur = queue.pop(0)
        if cur == '#':
            return None
        root = TreeNode(int(cur))
        root.left = self.process(queue)
        root.right = self.process(queue)

        return root

    def serialize1(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        # 后序
        if root is None:
            return '#_'

        leftRes = self.serialize(root.left)
        rightRes = self.serialize(root.right)

        # 返回值最后都有  _
        return leftRes + rightRes + str(root.val) + '_'

    def deserialize1(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        data1 = data.split('_')
        queue = []
        for x in data1:
            if x != '':  # 最后多个''
                queue.append(x)
        root = self.process1(queue)

        return root

    def process1(self, queue):
        cur = queue.pop()
        if cur == '#':
            return None
        root = TreeNode(int(cur))
        root.right = self.process1(queue)
        root.left = self.process1(queue)
        return root

# todo 貌似就是前后序方便，中序不好找root！！！    反复体会，关键啊！！！！



# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))