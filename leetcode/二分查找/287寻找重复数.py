class Solution:
    def findDuplicate1(self, nums) -> int:
        # todo 取值有范围的！要利用起来
        # todo n=5，取1-4，假设重复3，那1数量取值小于等于1，2取值小于等于2；而对于3，数量小于等于3的数量一定大于3，3之后的与3一样。
        # todo 下面的复杂度高，这种其实数量是有顺序的，可以二分！
        n = len(nums)

        mark = 0
        for i in range(1, n):
            num = 0
            for ele in nums:
                if ele <= i:
                    num += 1

            if num - mark <= 1:
                mark = num
                continue
            else:
                return i

    def findDuplicate(self, nums) -> int:
        l, r = 1, len(nums)

        while(l < r):
            mid = l + int((r - l) / 2)
            num = 0
            for ele in nums:
                if ele <= mid:
                    num += 1

            # mid是target及以后
            if num > mid:
                r = mid
            else:
                l = mid + 1

        return l

if __name__ == '__main__':
    s = Solution()
    res = s.findDuplicate([1,1])
    print(res)

# todo 如果l<r，那么mid可能等于l，但是r一定大于mid；l+1=r，那么mid==l。上述可以保证区域不断缩小！

# ok
# todo 思考二分的边界条件！