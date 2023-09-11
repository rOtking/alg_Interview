import functools


class Solution:
    # 1. 冒泡排序的应用
    # 虽然通过了，但是有点耗时
    def minNumber1(self, nums: List[int]) -> str:

        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if int(str(nums[j]) + str(nums[j + 1])) > int(str(nums[j + 1]) + str(nums[j])):
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]

        res = ''
        for x in nums:
            res += str(x)
        return res

    # 2. sorted的应用
    def minNumber(self, nums: List[int]) -> str:
        def cmp2key(a, b):
            if int(str(a) + str(b)) < int(str(b) + str(a)):
                return -1
            elif int(str(a) + str(b)) > int(str(b) + str(a)):
                return 1
            else:
                return 0

        nums = sorted(nums, key=functools.cmp_to_key(cmp2key))
        res = ''
        for x in nums:
            res += str(x)
        return res


# ok
# todo 其实就是一道排序！只不过交换的条件不是单纯的数的大小！