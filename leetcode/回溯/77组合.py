class Solution:
    def __init__(self):
        self.res = []
    def combine(self, n: int, k: int):
        candidates = [i for i in range(1, n + 1)]
        self.dfs(track=[], candidates=candidates, begin=0, k=k)
        return self.res

    def dfs(self, track, candidates, begin, k):
        if len(track) == k:
            self.res.append(track[:])
            return

        # todo
        # 恭喜自己：这是不重复candidates，结果变排列为组合的去重好方法！
        start = begin + 1
        for candidate in candidates[begin:]:
            if candidate in track:
                continue
            track.append(candidate)
            self.dfs(track, candidates, start, k)
            track.pop()
            start += 1

        return

if __name__ == '__main__':
    s = Solution()
    res = s.combine(4, 3)
    print(res)


# ok
# ez！哈哈
