# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0 or len(preorder) != len(inorder):
            return None

        root_val = preorder[0]
        # 确定 前序中左右分界点

        # root在中序遍历的位置
        root_in_index = inorder.index(root_val)
        # 左子树节点数量
        left_num = root_in_index

        # 前序中 左子树结束的index
        left_pre_end = left_num
        # 前序中 右子树开始的index
        right_pre_start = left_pre_end + 1

        left = self.buildTree(preorder[1:left_pre_end+1], inorder[0:left_num])
        right = self.buildTree(preorder[right_pre_start:], inorder[root_in_index + 1:])

        root = TreeNode(root_val, left=left, right=right)

        return root

# ok
# 关键点：通过左右子树的数量来确定位置！
# 前提是没有重复数字！！！！
    # 迭代法看不懂 算了吧