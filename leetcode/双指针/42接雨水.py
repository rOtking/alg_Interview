class Solution:
    def trap_dp(self, height: List[int]) -> int:
        # todo dp 貌似耗时空间的拉垮的很！
        # 先求leftMax与rightMax
        leftMax = [0] * len(height)
        rightMax = [0] * len(height)
        for i in range(len(height)):
            leftMax[i] = max(height[:i]) if i > 0 else 0
        for i in range(len(height)):
            rightMax[i] = max(height[i+1:]) if i < len(height) - 1 else 0

        res = 0
        for i in range(len(height)):
            # todo 注意负数就取0
            res += min(leftMax[i], rightMax[i]) - height[i] if min(leftMax[i], rightMax[i]) - height[i] > 0 else 0

        return res

    def trap1(self, height: List[int]) -> int:
        # 空间优化，但还不够！
        # todo 说白了，还是每次对数组算一次max，还是挺慢的
        res = 0
        for i in range(len(height)):
            leftMax = max(height[:i]) if i > 0 else 0
            rightMax = max(height[i + 1:]) if i < len(height) - 1 else 0
            res += min(leftMax, rightMax) - height[i] if min(leftMax, rightMax) - height[i] > 0 else 0

        return res

    def trap(self, height: List[int]) -> int:
        # 双指针优化
        leftMax, rightMax = 0, 0
        i, j = 0, len(height) - 1
        res = 0
        while(i <= j):
            # todo 这里提前更新很精髓！左边如果小于当前，那就让他等于当前，下面相见等于0，因为它本来就接不到水，还不会出现负数！
            # todo 左边大于当前，那还是左边的值；
            # todo 每次进循环前，leftMax都是i位置之前的最大值；执行后就是包含当前位置的最大值！
            # todo 一定牢记：只算当前位置的水，不要管之前！
            leftMax = max(leftMax, height[i])
            rightMax = max(rightMax, height[j])

            if height[i] < height[j]:
                # 那i的min一定来自左边，因为右边有一个已经明确比它大了
                res += leftMax - height[i]
                i += 1
            else:
                res += rightMax - height[j]
                j -= 1

        return res



# todo ok！一定反复看！！！！！   经典！






# todo 自己没想出来，好好看！核心思想是：考虑每个位置能接多少水，取决于min(左边最大高度, 右边最大高度)-当前感度
# todo 思想明确了，就简单了，重点就是怎么求每个位置左右最大值。
# todo 其实算dp问题，因为求leftmax与rightmax数组是用dp来做的！这其实是综合问题，不想单纯的dp问题，直接出结果，dp只是工具！很棒的题！
# todo dp的能从暴力递归来的，这题明显可以。
# todo 双指针其实是dp的空间优化问题，dp的更新如果只与前一个位置有关的话，那可以供滚动数组的思想来进行空间优化。