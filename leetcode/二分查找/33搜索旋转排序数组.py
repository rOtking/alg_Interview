class Solution:
    def search1(self, nums, target: int) -> int:
        # todo 这是弱智做法啊！O（n）还不够！

        index = -1
        for i, ele in enumerate(nums):
            if target == ele:
                index = i
                break

        return index

# todo 局部有序的数组还能用二分！只要有序就能二分！
# todo 旋转后，中间分开，一定一个有序一个无序！

    def search(self, nums, target: int) -> int:
        start = 0
        end = len(nums) - 1
        while(end >= start):
            mid = start + int((end - start) / 2)
            if nums[mid] == target:
                return mid
            # 左半有序
            if nums[start] <= nums[mid]:
                if nums[start] <= target <= nums[mid]:
                    end = mid
                else:
                    start = mid + 1
            # 右有序
            else:
                if nums[mid + 1] <= target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid

        return -1

if __name__ == '__main__':
    s = Solution()
    res = s.search(nums=[4,5,6,7,0,1,2], target=0)
    print(res)

# ok
# todo 部分有序也能二分！想办法找到局部有序二分！无序的再说。