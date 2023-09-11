class Solution:
    def __init__(self):
        self.res = []
        self.candidates_times = {}
    def combinationSum2(self, candidates, target: int):
        self.candidates_times = self.calTimes(candidates)
        candidates = sorted(list(set(candidates)))
        self.backtrack(track=[], candidates=candidates, target=target, begin=0)
        return self.res

    def backtrack(self, track, candidates, target, begin):
        if self.isOverflow(track, candidates):
            return
        if sum(track) == target:
            self.res.append(track[:])
            return

        start = begin
        for candidate in candidates[begin:]:
            track.append(candidate)
            self.backtrack(track, candidates, target, start)
            track.pop()
            start += 1

        return

    def calTimes(self, nums):
        # 计算频次
        times = {}
        for i in nums:
            if i in times:
                times[i] += 1
            else:
                times[i] = 1
        return times

    def isOverflow(self, track, candidates):
        '''
        判断track中的使用数量是否超了
        :param track:
        :param candidates:
        :return:
        '''
        track_times = self.calTimes(track)

        for k,v in track_times.items():
            if v > self.candidates_times[k]:
                return True

        return False



if __name__ == '__main__':
    s = Solution()
    res = s.combinationSum2([1], 2)
    print(res)


# todo 还不对，还是没理解，需要重新看！
