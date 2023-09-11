class Solution:
    def search(self, nums, target: int) -> bool:

        l, r = 0, len(nums) - 1
        while(l <= r):
            mid = l + int((r - l) / 2)
            if nums[mid] == target:
                return True
            # todo 不能把target删掉！
            # todo l的左边与r的右边可能相等，那是重复的数字，只要不是target，那就去掉他们！
            if nums[l] != target and nums[l] == nums[r]:
                l += 1
                r -= 1
            # 判断左右局部有序
            elif nums[l] <= nums[mid]:
                if nums[l] <= target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[r] >= target >= nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1

        return False


if __name__ == '__main__':
    s = Solution()
    res = s.search(nums=[1,2,1], target=2)
    print(res)


# ok
# todo 再看！都是细节！