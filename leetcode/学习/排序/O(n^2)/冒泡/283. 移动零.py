class Solution:
    def moveZeroes1(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] == 0 and nums[j + 1] != 0:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]

        return nums


    # ok ! 需要再考虑的！
    # todo

    def moveZeroes(self, nums: List[int]) -> None:
        # slow是依次找为0的位置，fast是slow右边第一个不为0的位置
        slow, fast = 0, 0
        while(slow < len(nums) and fast < len(nums)):
            if nums[slow] == 0:
                if nums[fast] != 0 and fast > slow:
                    nums[slow], nums[fast] = nums[fast], nums[slow]
                    fast += 1
                else:
                    fast += 1
            else:
                slow += 1
        return nums









# ok 还是稳定排序的应用，注意双指针会改变相对顺序，不能用。
