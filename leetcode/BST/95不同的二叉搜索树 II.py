# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # 构建中序遍历是nums的BST
    def process(self, nums):
        res = []
        if len(nums) == 0:
            return res
        if len(nums) == 1:
            res.append(TreeNode(nums[0]))
            return res
        # 每个位置都能做root
        for i, num in enumerate(nums):
            if i == 0:
                rights = self.process(nums[i+1:])
                for right in rights:
                    root = TreeNode(num)
                    root.right = right
                    res.append(root)
            elif i == len(nums) - 1:
                lefts = self.process(nums[:i])
                for left in lefts:
                    root = TreeNode(num)
                    root.left = left
                    res.append(root)
            else:
                lefts = self.process(nums[:i])
                rights = self.process(nums[i+1:])
                for left in lefts:
                    for right in rights:
                        root = TreeNode(num)
                        root.left = left
                        root.right = right
                        res.append(root)
        return res

    def generateTrees(self, n: int) -> List[TreeNode]:
        nums = [i for i in range(1, n + 1)]
        return self.process(nums)

    # 答案所谓的DFS和自己想的是一样的！！！就是递归



if __name__ == '__main__':
    s = Solution()
    res = s.generateTrees(3)
    print(res)


# ok
# BST分左右更容易，一定利用好！todo