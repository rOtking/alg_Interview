import copy

class Solution:
    def __init__(self):
        self.res = []
    def permuteUnique(self, nums):
        self.backtrack(track=[], nums=nums)
        return self.res

    def backtrack(self, track, nums):
        '''
        :param track:
        :param nums: todo 这次是可选择列表，变一下试试
        :return:
        '''
        # if len(nums) == 0 or self.isSame(nums):
        if len(nums) == 0:
            tmp = copy.copy(track)
            # if self.isSame(nums):
            #     tmp.extend(nums)
            self.res.append(tmp)
            return

        # todo 关键 做不重复的选择
        todo = set(nums)
        for num in todo:
            track.append(num)
            # 新的待选择列表
            new_nums = copy.copy(nums)
            new_nums.remove(num)
            self.backtrack(track, new_nums)
            track.pop()
        return

    # def isSame(self, nums):
    #     '''
    #     判断是否都是同一个数字
    #     :param nums:
    #     :return: bool
    #     '''
    #     res = True
    #     key = nums[0]
    #     for num in nums:
    #         if num != key:
    #             res = False
    #             break
    #     return res



if __name__ == '__main__':
    nums = [1,1,2]
    s = Solution()
    res = s.permuteUnique(nums)
    print(res)



# 数字可重复
# 用决策树模拟
# 终止条件加了一点：可选择的选项都一样时，就可以提前停止，算为一种！

# ok
# 想好决策树，每次选择都是不重复的选择 set