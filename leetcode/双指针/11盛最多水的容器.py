class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        res_l, res_r = 0, len(height) - 1
        area = 0
        while(left < right):
            area_cur = (right - left) * min(height[left], height[right])
            if area < area_cur:
                res_l, res_r = left, right
                area = area_cur
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return area



# todo 盛水问题还是双指针好用，并且有贪心的思想。双指针 + 贪心：控制指针移动是个标准套路。
# ok 自己做出来了！
# todo 核心：每次都移动小的，因为当前area取决于小的*距离，如果移动大的，距离一定变小，而最小值只会不变或变小，area只会越来越小！
# todo     如果移动小的，距离虽然小了，但是height的最小值是可能变大的，area的潜在解一定在这种情况下！