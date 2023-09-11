class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # 因为只要一个满足的峰值即可，所以可以二分！
        # todo 核心想法：mid下降，说明峰值在左，不可能没有！那就去左边找；mid上升，右边；mid峰值；
        # 数不同，没有等于的情况
        # todo 边界值也可以是峰值！mid是低谷说明两侧都有上升趋势，所以两边一定都有峰值，找一个即可！

        # 因为n-1要处理边界，那就只考虑n+1
        l, r = 0, len(nums) - 1

        # 为了不让n+1越界 不是<=
        while(l < r):
            mid = l + int((r - l) / 2)
            if nums[mid] > nums[mid + 1]:
                # 右边下降，左边可能升降，【l，r】是范围，闭合的
                r = mid
            else:
                l = mid + 1
        # 此时l == r
        # 最终的范围就是一个数，且一定有一个数！
        return l







# ok
# todo 这个题必须理解题意，就很简单了！不要自己臆想题目的意图！



