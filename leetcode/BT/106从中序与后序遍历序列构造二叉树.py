# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, inorder, postorder):
        if len(inorder) == 0 or len(postorder) == 0:
            return None
        assert len(inorder) == len(postorder), '中序后序列表长度不一致！'

        root_val = postorder[-1]
        # 根节点在中序的index
        root_in_index = inorder.index(root_val)
        # 左子树节点个数
        left_num = root_in_index

        # 后序中左子树的结束位置
        left_post_end = left_num - 1
        # 后序中右子树的开始位置
        right_post_start = left_post_end + 1

        # todo 重点：后序遍历框架！

        left = self.buildTree(inorder[:root_in_index], postorder[:left_post_end + 1])
        right = self.buildTree(inorder[root_in_index + 1:], postorder[right_post_start:-1])

        root = TreeNode(root_val, left=left, right=right)

        return root

if __name__ == '__main__':
    inorder = [9,3,15,20,7]
    postorder = [9,15,7,20,3]

    s = Solution()
    root = s.buildTree(inorder, postorder)


# ok

# 一样的，很简单，但是浪费了不少时间！
# 因为索引 +1 还是-1 的细节要谨慎！