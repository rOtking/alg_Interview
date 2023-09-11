# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # 可以递归的做；但是
        # todo 求最短路径，显然是BFS的完美使用！BFS的思想，随着框架模拟一遍就很清晰了！
        if TreeNode is None:
            return 0
        queue = [root]
        visited = set(root)

        depth = 1
        while(len(queue) != 0):
            for _ in range(len(queue)):
                cur = queue.pop(0)
                if cur.left is None and cur.right is None:
                    return depth

                if cur.left is not None and cur.left not in visited:
                    queue.append(cur.left)
                    visited.add(cur.left)

                if cur.right is not None and cur.right not in visited:
                    queue.append(cur.right)
                    visited.add(cur.right)
            depth += 1


if __name__ == '__main__':
    pass

# ok
# todo leecode得分比递归高多了 哈哈BFS果然是最短路径问题的神！
