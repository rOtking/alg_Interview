class Solution:
    def findMin(self, nums) -> int:
        l, r = 0, len(nums) - 1
        while(l < r):
            mid = l + int((r - l) / 2)
            # 不需要！
            # if nums[mid - 1] > nums[mid] < nums[mid + 1]:
            #     return nums[mid]

            # 最小一定在无序区；也有可能本来就是完全有序的！
            # todo 直接看右边，右边有序的话，无论左边有没有序，结果都在左边！
            # 在左
            if nums[mid] < nums[r]:
                r = mid     # mid可能是
            else:
                l = mid + 1

        return nums[l]



if __name__ == '__main__':
    s = Solution()
    res = s.findMin(nums=[11,13,15,17])
    print(res)

# ok
# todo 继续看，全是细节！
