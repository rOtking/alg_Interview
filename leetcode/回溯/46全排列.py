import copy
class Solution:
    def __init__(self):
        self.res = []
    def permute(self, nums):
        self.backtrack(track=[], nums=nums)
        return self.res

    def backtrack(self, track, nums):
        '''
        回溯
        :param track: 已做过的选择
        :param nums: 所有选择
        :return:
        '''

        # 终止条件
        if len(track) == len(nums):
            # list的赋值会改变，copy才行
            self.res.append(copy.copy(track))
            return

        for num in nums:
            # 处理选择列表的
            if num in track:
                continue
            track.append(num)
            self.backtrack(track, nums)
            track.pop()
        return

if __name__ == '__main__':
    nums = [1,2,3]
    s = Solution()
    res = s.permute(nums)
    print(res)


# ok
# todo 记住框架！