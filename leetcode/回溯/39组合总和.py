import copy

class Solution:
    def __init__(self):
        self.res = []

    def combinationSum(self, candidates, target: int):
        candidates = sorted(candidates)
        self.backtrack(track=[], candidates=candidates, target=target, begin=0)
        return self.res

    def backtrack(self, track, candidates, target, begin):
        if sum(track) == target:
            tmp = copy.copy(track)
            self.res.append(tmp)
            return
        # 大了就直接终止
        if sum(track) > target:
            return
        # todo begin与start是关键。不同排序算一种组合的去重问题：就是对选择列表进行限制！
        # begin是当前能选择的，start
        # 如2开头已经选过了，说明与2有关的在之前全选了；现在从3开始，就不会再选2
        # 剪枝！
        start = begin
        for candidate in candidates[begin:]:
            track.append(candidate)
            self.backtrack(track, candidates, target, start)
            track.pop()
            start += 1
        return

if __name__ == '__main__':
    s = Solution()
    candidates = [2,3,6,7]
    res = s.combinationSum(candidates, 7)
    print(res)


# ok
# todo 回溯的去重问题！可以全输出在用dict去重。上面是在选择的时候剪枝，更快。