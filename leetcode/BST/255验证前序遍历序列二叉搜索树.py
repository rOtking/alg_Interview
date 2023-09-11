class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        if len(preorder) == 0:
            return True
        if len(preorder) == 1:
            return True
        # 按BST特征找左右子节点
        # 求0位置右边最近的比他大的数
        i = 1
        while (i < len(preorder) and preorder[i] < preorder[0]):
            i += 1
        if i == len(preorder) or i == 1:
            # 没有比0位置大的 或 没有小的 执行的都一样
            return self.verifyPreorder(preorder[1:])
        else:
            return self.verifyPreorder(preorder[1:i]) and self.verifyPreorder(preorder[i:])

    # [5,2,6,1,3] 不对

    # todo 单调栈！！！！先不管了，看不懂，小众。



